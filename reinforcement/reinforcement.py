# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs a reinforcement learning loop to train a Go playing model."""
import logging
import os
import re
import subprocess
import sys
import shutil

import dual_net
import evaluate
import strategies
import utils

from absl import app, flags
from tensorflow import gfile

from rl_loop import example_buffer
from rl_loop import fsdb
from rl_loop import shipname

flags.adopt_module_key_flags(dual_net)
flags.adopt_module_key_flags(evaluate)
flags.adopt_module_key_flags(strategies)

flags.DEFINE_string('engine', 'tf', 'Engine to use for inference.')

FLAGS = flags.FLAGS


def checked_run(cmd, name):
  logging.info('Running %s:\n  %s', name, '\n  '.join(cmd))
  with utils.logged_timer('%s finished' % name.capitalize()):
    completed_process = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert completed_process.returncode == 0, \
      completed_process.stdout.decode()
  return completed_process


def get_lines(completed_process, slice):
  return '\n'.join(completed_process.stdout.decode()[:-1].split('\n')[slice])


class MakeSlice(object):

  def __getitem__(self, item):
    return item


make_slice = MakeSlice()


def play_model_path(model):
  return os.path.join(fsdb.models_dir(), model.play_model_name)


def train_model_path(model):
  return os.path.join(fsdb.models_dir(), model.train_model_name)


def cc_flags(model):
  return [
      '--engine={}'.format(FLAGS.engine),
      '--virtual_losses={}'.format(FLAGS.parallel_readouts),
      '--num_readouts={}'.format(FLAGS.num_readouts),
      '--seed={}'.format(model.model_num + 1),
  ]


def bootstrap(model):
  checked_run([
      'python3', 'external/minigo/bootstrap.py', '--export_path={}'.format(
          play_model_path(model)), '--work_dir={}'.format(fsdb.working_dir())
  ], 'bootstrap')


# Self-play a number of games.
def selfplay(model):
  play_output_name = '%06d%s' % (model.model_num, model.play_model_name[6:])
  play_output_path = os.path.join(fsdb.selfplay_dir(), play_output_name)

  numselfplay_games = 2048
  result = checked_run([
      'external/minigo/cc/main', '--mode=selfplay', '--parallel_games={}'
      .format(numselfplay_games), '--model={}'.format(play_model_path(model)),
      '--output_dir={}'.format(play_output_path), '--holdout_pct=0.0'
  ] + cc_flags(model), 'selfplay')
  logging.info(get_lines(result, make_slice[-2:]))

  # Write examples to a single record.
  logging.info('Extracting examples')
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
  buffer.parallel_fill(gfile.Glob(os.path.join(play_output_path, '*/*.zz')))
  buffer.flush(
      os.path.join(fsdb.golden_chunk_dir(), play_output_name + '.tfrecord.zz'))


# Train a new model.
def train(model):
  # Train on shuffled game data of the last 10 selfplay rounds.
  tf_records = os.path.join(fsdb.golden_chunk_dir(), '*.tfrecord.zz')
  tf_records = sorted(gfile.Glob(tf_records), reverse=True)[:10]
  result = checked_run([
      'python3',
      'external/minigo/train.py',
      *tf_records,
      '--value_cost_weight=0.5',  # TODO(csigg): what value?
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(train_model_path(model)),
  ], 'training')
  logging.info(get_lines(result, make_slice[-8:-8]))


# Evaluate the trained model.
def evaluate(model, opponent_arg, name, slice):
  sgf_dir = os.path.join(fsdb.eval_dir(), model.train_model_name)
  result = checked_run([
      'external/minigo/cc/main', '--mode=eval',
      '--parallel_games={}'.format(100), '--model={}'.format(
          train_model_path(model)), opponent_arg, '--sgf_dir={}'.format(sgf_dir)
  ] + cc_flags(model), name)
  result = get_lines(result, slice)
  logging.info(result)
  pattern = '{}\s+\d+\s+(\d+\.\d+)%'.format(model.train_model_name)
  return float(re.search(pattern, result).group(1)) * 0.01


# Evaluate trained model against previous best.
def evaluate_model(model):
  model_win_rate = evaluate(model, '--model_two={}'.format(
      play_model_path(model)), 'model eval', make_slice[-7:])
  logging.info('Win rate %s vs %s: %.3f', model.train_model_name,
               model.play_model_name, model_win_rate)
  return model_win_rate


# Evaluate trained model against other Go program.
def evaluate_target(model):
  leela_cmd = 'external/leela/leela_0110_linux_x64 ' \
              '--gtp --quiet --playouts=2000 --noponder'
  # gnugo_cmd = '/usr/games/gnugo --mode gtp --chinese-rules --level 10'
  target_win_rate = evaluate(model, '--gtp_client={}'.format(leela_cmd),
                             'target eval', make_slice[-6:])
  logging.info('Win rate against %s vs Leela: %.3f', model.train_model_name,
               target_win_rate)
  return target_win_rate


# Play puzzles with the trained model.
def puzzle(model):
  result = checked_run([
      'external/minigo/cc/main', '--mode=puzzle', '--model={}'.format(
          train_model_path(model)), '--sgf_dir={}'.format('puzzles/')
  ] + cc_flags(model), 'puzzle')
  result = get_lines(result, make_slice[-2:])
  logging.info(result)
  pattern = 'Solved \d+ of \d+ puzzles \((\d+\.\d+)%\)'
  puzzle_prediction_rate = float(re.search(pattern, result).group(1)) * 0.01
  logging.info('Prediction rate of puzzles: %.3f', puzzle_prediction_rate)


class RlLoop(object):

  def __init__(self):
    self.model_num = 0
    self.model_win_rate = 0.0
    self.play_model_name = shipname.generate(0)
    self.train_model_name = shipname.generate(1)

    bootstrap(self)
    selfplay(self)

  def __iter__(self):
    return self

  def __next__(self):
    self.model_num = self.model_num + 1

    if self.model_win_rate >= 0.55:
      # Promote the trained model to the play model.
      self.play_model_name = self.train_model_name
      self.train_model_name = shipname.generate(self.model_num)
    else:
      # The trained model is not significantly better. Generate more
      # games with the same play model and train a new candidate.
      self.train_model_name = '%06d%s' % (self.model_num,
                                          self.train_model_name[6:])

    train(self)
    self.model_win_rate = evaluate_model(self)
    target_win_rate = evaluate_target(self)
    puzzle(self)

    # This could run in parallel with the block above.
    selfplay(self)

    return self.model_num, target_win_rate


def main(unused_argv):
  """Run the reinforcement learning loop."""

  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  shutil.rmtree(FLAGS.base_dir, ignore_errors=True)

  utils.ensure_dir_exists(fsdb.models_dir())
  utils.ensure_dir_exists(fsdb.selfplay_dir())
  utils.ensure_dir_exists(fsdb.eval_dir())
  utils.ensure_dir_exists(fsdb.golden_chunk_dir())
  utils.ensure_dir_exists(fsdb.working_dir())

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'reinforcement.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  for model_num, target_win_rate in RlLoop():
    if target_win_rate >= 42:  # TODO(csigg): Choose exit criteria
      return logging.info('Done')
    if model_num >= 100:
      return logging.info('Failed to converge')


if __name__ == '__main__':
  sys.path.insert(0, '.')
  with utils.logged_timer('Total time'):
    app.run(main)
