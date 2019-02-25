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
import numpy
import random
import re
import subprocess
import sys
import shutil
import tensorflow
import utils
import multiprocessing
import fcntl
import glob
import threading
import time

from absl import app, flags
from rl_loop import example_buffer, fsdb, shipname

flags.DEFINE_string('engine', 'tf', 'Engine to use for inference.')
flags.DEFINE_string('device', 'gpu', 'Device to use for inference.')

FLAGS = flags.FLAGS


# num_instance: number of instances totally launched
#               if num_instance == 0, go through the simple path with out any
#                   core affinity control
#               if num_instance >  0, go through the multi-instance path with
#                   core affinity control
# num_parallel_instance: number of instances running in parallel
# cores_per_instance: number of cores for one instance
def checked_run(cmd, name, num_instance=0, num_parallel_instance=None,
                cores_per_instance=1):
  if (num_instance == 0):
    logging.info('Running %s:\n  %s', name, '\n  '.join(cmd))
  else:
    logging.info('Running %s*%d:\n  %s', name, num_instance, '\n  '.join(cmd))
  with utils.logged_timer('%s finished' % name.capitalize()):
    # if num_instance == 0, use default behavior for GPU
    if num_instance == 0:
      try:
        cmd = ' '.join(cmd)
        completed_output = subprocess.check_output(
          cmd, shell=True, stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as err:
        logging.error('Error running %s: %s', name, err.output.decode())
        raise RuntimeError('Non-zero return code executing %s' % ' '.join(cmd))
      return completed_output
    elif num_instance == 1:
      if cores_per_instance != None:
        omp_string = 'OMP_NUM_THREADSD={}'.format(cores_per_instance)
      else:
        omp_string = ''
      prefix =' '.join([
          omp_string,
          'KMP_AFFINITY=compact,granularity=fine,1,0'])
      try:
        cmd = prefix + ' '  + ' '.join(cmd)
        completed_output = subprocess.check_output(
          cmd, shell=True, stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as err:
        logging.error('Error running %s: %s', name, err.output.decode())
        raise RuntimeError('Non-zero return code executing %s' % ' '.join(cmd))
      logging.info(completed_output.decode())
      return completed_output
    else:
      if num_parallel_instance == None:
            num_parallel_instance = int(multiprocessing.cpu_count())
      elif num_parallel_instance <= 0:
            num_parallel_instance = int(multiprocessing.cpu_count()+num_parallel_instance)
      procs=[None]*num_parallel_instance
      results = [""]*num_parallel_instance
      lines = [""]*num_parallel_instance
      result=""

      cur_instance = 0
      # add new proc into procs
      while cur_instance < num_instance or not all (
          proc is None for proc in procs):
        if None in procs and cur_instance < num_instance:
          index = procs.index(None)
          subproc_cmd = [
                  'OMP_NUM_THREADS={}'.format(cores_per_instance),
                  'KMP_AFFINITY=granularity=fine,proclist=[{}],explicit'.format(
                      ','.join(str(i) for i in list(range(
                          index, index+cores_per_instance
                          ))))]
          subproc_cmd = subproc_cmd + cmd
          subproc_cmd = ' '.join(subproc_cmd)
          subproc_cmd = subproc_cmd.format(cur_instance)
          if (cur_instance == 0):
            logging.debug("subproc_cmd = {}".format(subproc_cmd))
          procs[index] = subprocess.Popen(subproc_cmd, shell=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT)

          proc_count = 0
          for i in range(num_parallel_instance):
            if procs[i] != None:
              proc_count += 1
          logging.debug('started instance {} in proc {}. proc count = {}'.format(
              cur_instance, index, proc_count))

          # change stdout of the process to non-blocking
          # this is for collect output in a single thread
          flags = fcntl.fcntl(procs[index].stdout, fcntl.F_GETFL)
          fcntl.fcntl(procs[index].stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

          cur_instance += 1
        for index in range(num_parallel_instance):
          if procs[index] != None:
            # collect proc output
            while True:
              try:
                line = procs[index].stdout.readline()
                if line == b'':
                  break
                results[index] = results[index] + line.decode()
              except IOError:
                break

            ret_val = procs[index].poll()
            if ret_val == None:
              continue
            elif ret_val != 0:
              logging.debug(results[index])
              raise RuntimeError(
                'Non-zero return code (%d) executing %s' % (
                    ret_val, subproc_cmd))

            result += results[index]
            results[index] = ""
            procs[index] = None

            proc_count = 0
            for i in range(num_parallel_instance):
              if procs[i] != None:
                proc_count += 1
            logging.debug('proc {} finished. proc count = {}'.format(
                index, proc_count))
        time.sleep(0.001)  # avoid busy loop
      return result.encode('utf-8')


def get_lines(completed_output, slice):
  return '\n'.join(completed_output.decode()[:-1].split('\n')[slice])


class MakeSlice(object):

  def __getitem__(self, item):
    return item


make_slice = MakeSlice()


def cc_flags(state):
  return [
      '--engine={}'.format(FLAGS.engine),
      '--virtual_losses=8',
      '--seed={}'.format(state.seed),
  ]


def py_flags(state):
  return [
      '--work_dir={}'.format(fsdb.working_dir()),
      '--training_seed={}'.format(state.seed),
  ]


# Generate an initial model with random weights.
def bootstrap(state):
  checked_run([
      'python3', 'external/minigo/bootstrap.py', '--export_path={}'.format(
          state.play_model_path)
  ] + py_flags(state), 'bootstrap')


# Self-play a number of games.
def selfplay(state, parallel_games=None, total_games=2048, num_parallel_instance=None, gen_sgf=False):
  if parallel_games == None:
    if FLAGS.device == 'cpu':
      parallel_games = 4
    else:
      parallel_games = total_games
  play_output_name = state.play_output_name
  play_output_dir = os.path.join(fsdb.selfplay_dir(), play_output_name)
  play_holdout_dir = os.path.join(fsdb.holdout_dir(), play_output_name)
  play_sgf_dir = os.path.join(fsdb.sgf_dir(), play_output_name)

  run_cmds = [
    'external/minigo/cc/main',
    '--mode=selfplay',
    '--parallel_games={}'.format(parallel_games),
    '--num_readouts=100',
    '--model={}'.format(state.play_model_path),
    '--output_dir={}'.format(play_output_dir),
    '--holdout_dir={}'.format(play_holdout_dir)
  ]
  if gen_sgf:
    run_cmds = run_cmds + ['--sgf_dir={}'.format(play_sgf_dir)]
  if (parallel_games == total_games):
    result = checked_run(run_cmds + cc_flags(state), 'selfplay')
  else:
    run_cmds = run_cmds + ['--instance_id={}']
    result = checked_run(run_cmds + cc_flags(state), 'selfplay',
                         num_instance=total_games/parallel_games, num_parallel_instance=num_parallel_instance)
  logging.info(get_lines(result, make_slice[-2:]))

  # Write examples to a single record.
  logging.info('Extracting examples')
  random.seed(state.seed)
  tensorflow.set_random_seed(state.seed)
  numpy.random.seed(state.seed)
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
  buffer.parallel_fill(
      tensorflow.gfile.Glob(os.path.join(play_output_dir, '*.zz')))
  buffer.flush(
      os.path.join(fsdb.golden_chunk_dir(), play_output_name + '.tfrecord.zz'))
  if gen_sgf:
      sgf_path = os.path.join(play_sgf_dir, "clean/*.sgf")
      sgf_files = glob.glob(sgf_path)
      full_sgf_path = os.path.join(play_sgf_dir, "full")
      white_win = 0
      black_win = 0
      for file_name in sgf_files:
        if 'W+' in open (file_name).read():
            white_win += 1
        if 'B+' in open (file_name).read():
            black_win += 1
      logging.info ("White win {} times, black win {} times.".format(white_win, black_win))
      bias = abs(white_win - black_win)/(white_win+black_win)
      logging.info ("selfplay bias = {}".format(bias))
      # remove full_sgf_files to save space
      with utils.logged_timer('remove sgf files'):
        logging.info('removing {}'.format(full_sgf_path))
        shutil.rmtree(full_sgf_path, ignore_errors=True)
      return bias

# Train a new model.
def train(state, tf_records, train_batch_size=256):
  if FLAGS.device == 'cpu':
    num_instance = 1
    num_parallel_instance=1
    cores_per_instance=56
  else:
    num_instance = 0
    num_parallel_instance=None
    cores_per_instance=None

  result = checked_run([
      'python3',
      'external/minigo/train.py',
      ] + tf_records + [
      '--train_batch_size={}'.format(train_batch_size),
      '--lr_rates={}'.format(0.01*train_batch_size/256),
      '--lr_rates={}'.format(0.001*train_batch_size/256),
      '--lr_rates={}'.format(0.0001*train_batch_size/256),
      '--lr_boundaries={}'.format(int(400000*256/train_batch_size)),
      '--lr_boundaries={}'.format(int(600000*256/train_batch_size)),
      '--export_path={}'.format(state.train_model_path),
  ] + py_flags(state), 'training', num_instance, num_parallel_instance, cores_per_instance)
  logging.info(get_lines(result, make_slice[-8:-8]))


# Validate the trained model against holdout games.
def validate(state, holdout_dir):
  result = checked_run(
      ['python3', 'external/minigo/validate.py', holdout_dir] + py_flags(state),
      'validation')
  logging.info(get_lines(result, make_slice[-4:-3]))


# Evaluate the trained model.
def evaluate(state, args, name, slice):
  sgf_dir = os.path.join(fsdb.eval_dir(), state.train_model_name)
  if FLAGS.device == 'cpu':
    result = checked_run([
        'external/minigo/cc/main', '--mode=eval', '--parallel_games=2',
        '--instance_id={}',
        '--model={}'.format(
            state.train_model_path), '--sgf_dir={}'.format(sgf_dir)
    ] + args, name, num_instance=50)
  else:
    result = checked_run([
        'external/minigo/cc/main', '--mode=eval', '--parallel_games=100',
        '--model={}'.format(
            state.train_model_path), '--sgf_dir={}'.format(sgf_dir)
    ] + args, name)

  result = result.decode()
  logging.info(result)
  pattern = '{}\s+(\d+)\s+\d+\.\d+%\s+(\d+)\s+\d+\.\d+%\s+(\d+)'.format(
                state.train_model_name)
  matches = re.findall(pattern, result)
  total = 0.0
  black = 0.0
  white = 0.0
  for i in range(len(matches)):
    total += float(matches[i][0])
    black += float(matches[i][1])
    white += float(matches[i][2])
  return total * 0.01, black * 0.02, white * 0.02


# Evaluate trained model against previous best.
def evaluate_model(state):
  model_win_rate, black_win_rate, white_win_rate = evaluate(
      state,
      ['--num_readouts=100', '--model_two={}'.format(state.play_model_path)
      ] + cc_flags(state), 'model evaluation', make_slice[-7:])
  logging.info('Win rate %s vs %s: %.3f', state.train_model_name,
               state.play_model_name, model_win_rate)
  logging.info('Black win rate %.3f, white win rate %.3f',
               black_win_rate, white_win_rate)
  if black_win_rate + white_win_rate > 0:
    bias = abs(black_win_rate - white_win_rate) / (black_win_rate + white_win_rate)
  else:
    bias = 0.5
  logging.info ("evaluate win rate = {} , bias = {}".format(model_win_rate, bias))
  return model_win_rate, bias

# Evaluate trained model against reference minigo model.
def evaluate_target_minigo(state):
  target_win_rate, _, _ = evaluate(
      state,
      ['--num_readouts=100',
       '--model_two={}'.format(
          'target-model/fifty-fifty.pb')
      ] + cc_flags(state), 'target evaluation', make_slice[-7:])
  logging.info('Win rate %s vs target: %.3f', state.train_model_name,
               target_win_rate)
  return target_win_rate

# Evaluate trained model against Leela.
def evaluate_target_leela(state):
  leela_cmd = '"external/leela/leela_0110_linux_x64 ' \
              '--gtp --quiet --playouts=2000 --noponder"'
  target_win_rate, _, _ = evaluate(
      state, ['--num_readouts=400', '--gtp_client={}'.format(leela_cmd)
             ] + cc_flags(state), 'target evaluation', make_slice[-6:])
  logging.info('Win rate  %s vs Leela: %.3f', state.train_model_name,
               target_win_rate)
  return target_win_rate

def evaluate_target(state):
    return evaluate_target_minigo(state)

class State:

  _NAMES = ['bootstrap'] + random.Random(0).sample(shipname.NAMES,
                                                   len(shipname.NAMES))

  def __init__(self):
    self.iter_num = 0
    self.play_model_num = 0
    self.play_model_name = self.play_output_name
    self.train_model_num = 1

  @property
  def play_output_name(self):
    return '%06d-%s' % (self.iter_num, self._NAMES[self.play_model_num])

  @property
  def play_model_path(self):
    return os.path.join(fsdb.models_dir(), self.play_model_name)

  @property
  def train_model_name(self):
    return '%06d-%s' % (self.iter_num, self._NAMES[self.train_model_num])

  @property
  def train_model_path(self):
    return os.path.join(fsdb.models_dir(), self.train_model_name)

  @property
  def seed(self):
    return self.iter_num + 1


class func_thread(threading.Thread):
  def __init__(self, func, *args):
    threading.Thread.__init__(self)
    self._target = func
    self._args   = args

  def run(self):
    self._target(*self._args)

def rl_loop(base_name):
  max_iter = 104
  train_set_count_max = 5
  train_set_count = 5
  # Generate 4 bootstrap models
  state = State()
  bootstrap_count = 5
  for index in range (bootstrap_count):
    while True:
      bootstrap(state)
      bias = selfplay(state, gen_sgf=True)
      if bias < 0.5:
        if index < bootstrap_count-1:
          state.iter_num += 1
          state.play_model_name = state.play_output_name
        break
      state.iter_num += 1
      logging.info ("skip bootstrap with bias: {}".format(bias))
      tf_records = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
      tf_records = sorted(tensorflow.gfile.Glob(tf_records), reverse=True)[:1]
      os.remove(tf_records[0])
      for f in glob.glob(os.path.join(fsdb.models_dir(), state.play_model_name+'*')):
        os.remove(f)
      shutil.rmtree(os.path.join(fsdb.selfplay_dir(), state.play_model_name),
                    ignore_errors=True)
      shutil.rmtree(os.path.join(fsdb.sgf_dir(), state.play_model_name),
                    ignore_errors=True)
      shutil.rmtree(os.path.join(fsdb.holdout_dir(), state.play_model_name),
                    ignore_errors=True)

  while state.iter_num < max_iter:
    train_set_count += 1
    if train_set_count > train_set_count_max:
      train_set_count = train_set_count_max
    holdout_dir = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num)
    tf_records = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
    tf_records = sorted(tensorflow.gfile.Glob(tf_records),
                        reverse=True)[:train_set_count]

    state.iter_num += 1
    # Train on shuffled game data of the last 5 selfplay rounds.
    train(state, tf_records)

    validate(state, holdout_dir)
    model_win_rate, bias = evaluate_model(state)
    target_win_rate = evaluate_target(state)

    if model_win_rate >= 0.55:
      # a promotion candidate, check whether its high biased
      temp_play_model_num = state.play_model_num
      temp_play_model_name = state.play_model_name
      state.play_model_num = state.train_model_num
      state.play_model_name = state.train_model_name
      bias = selfplay(state, gen_sgf=True)
      if bias > 0.7:
        # Do not promote if trained model has high bias
        state.play_model_num = temp_play_model_num
        state.play_model_name = temp_play_model_name
        # Also abandom this selfplay result
        tf_records = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
        tf_records = sorted(tensorflow.gfile.Glob(tf_records), reverse=True)[:1]
        logging.info('Burying %s for selfplay bias >0.7.', tf_records[0])
        shutil.move(tf_records[0], tf_records[0] + '.bury')
        bias = selfplay(state, gen_sgf=True)
      else:
        # Promote the trained model to the play model.
        state.play_model_num = state.train_model_num
        state.play_model_name = state.train_model_name
        state.train_model_num += 1
        # if promoted model is significantly better, old selfplay has no use
        if model_win_rate >= 0.8:
          train_set_count = 0
    else:
      if model_win_rate < 0.4:
        # Randomly bury one selfplay game set which produced a significantly
        # worse model.
        bury_index = random.randint(0, len(tf_records)-1)
        logging.info('Burying %s for model win rate < 0.4.',
                     tf_records[bury_index])
        shutil.move(tf_records[bury_index], tf_records[bury_index] + '.bury')
      bias = selfplay(state, gen_sgf=True)

    yield target_win_rate

def wipe_dir(base_dir):
  print('Wiping dir %s' % base_dir, flush=True)
  shutil.rmtree(base_dir, ignore_errors=True)

  utils.ensure_dir_exists(fsdb.models_dir())
  utils.ensure_dir_exists(fsdb.selfplay_dir())
  utils.ensure_dir_exists(fsdb.holdout_dir())
  utils.ensure_dir_exists(fsdb.eval_dir())
  utils.ensure_dir_exists(fsdb.golden_chunk_dir())
  utils.ensure_dir_exists(fsdb.working_dir())

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'reinforcement.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

def main(unused_argv):
  """Run the reinforcement learning loop."""

  wipe_dir(FLAGS.base_dir)
  with utils.logged_timer('Total time'):
    for target_win_rate in rl_loop(FLAGS.base_dir):
      if target_win_rate > 0.5:
        return logging.info('Passed exit criteria.')
    logging.info('Failed to converge.')


if __name__ == '__main__':
  sys.path.insert(0, '.')
  app.run(main)
