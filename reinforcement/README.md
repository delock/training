# <span style="color:red">Preliminary</span>
This repo is a staging ground for the new [MLPerf](http://mlperf.org) reinforcement model.
Eventually this code will replace the code in
[this directory](http://github.com/mlperf/training/tree/master/reinforcement/tensorflow/minigo).

# 1. Problem
This task benchmarks on policy reinforcement learning for the 9x9 version of the boardgame Go.
The model plays games against itself and uses these games to improve play.

# 2. Directions
### Steps to configure machine
To setup the environment on Ubuntu 16.04 (16 CPUs, one P100, 100 GB disk), you can use these
commands. This may vary on a different operating system or graphics card.

    # Clone repository
    git clone git@github.com:chsigg/training.git
    # Note: This will eventually change to:
    # git clone git@github.com:mlperf/training.git
    cd training/reinforcement

    # Install dependencies
    apt-get install -y python3 python3-pip rsync git wget pkg-config zip g++ zlib1g-dev unzip

    # Install python dependencies
    pip3 install -r requirements.txt

    # Install bazel
    wget https://github.com/bazelbuild/bazel/releases/download/0.17.1/bazel-0.17.1-installer-linux-x86_64.sh
    chmod +x bazel-0.17.1-installer-linux-x86_64.sh
    ./bazel-0.17.1-installer-linux-x86_64.sh

    # Compile and run
    bazel run //:reinforcement -- --base_dir=$(pwd)/results/$(date +%Y-%m-%d)


### Alternatively, use Docker
To run the reinforcement benchmark in a Docker container, you can use these commands instead.

    # Build the docker
    docker build -t reinforcement .

    # Run the benchmark inside docker
    docker run --runtime=nvidia -v $(pwd)/results:/results reinforcement


### Steps to download and verify data
Unlike other benchmarks, there is no data to download. All training data comes from games played
during benchmarking.

# 3. Model
### Publication/Attribution

This benchmark is based on the minigo project (https://github.com/tensorflow/minigo); which is
inspired by the work done by Deepmind with ["Mastering the Game of Go with Deep Neural Networks and
Tree Search"](https://www.nature.com/articles/nature16961), ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270), and ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815).
Note that MiniGo is an independent effort from AlphaGo, and that this fork is minigo is independent
from minigo itself.

### Reinforcement Setup
This benchmark includes both the environment and training for 9x9 Go. There are four primary phases
in this benchmark, these phases are repeated in order:

 - Selfplay: the *current best* model plays games against itself to produce board positions for
   training.
 - Training: train the neural networks using selfplay data from recent models. We are not using the
   selfplay data from the current iteration so that selfplay and the rest of the iteration may run
   in parallel.
 - Model Evaluation: the *current best* and the most recently trained model play a series of games.
   In order to become the new *current best*, the most recently trained model must win 55% of the
   games.
 - Target Evaluation: if the newly trained model has been promoted to the current best, play a series
   of games against [Leela](https://sjeng.org/leela.html). The termination criteria for the benchmark
   is to win at least 50% of the games.

### Structure
This task has a non-trivial network structure, including a search tree. A good overview of the
structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0.

### Weight and bias initialization and Loss Function
Network weights are initialized randomly. Initialization and loss are described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the primary network.

# 4. Quality
Due to the difficulty of training a highly proficient Go model, our quality metric and termination
criteria is based on winning against another Go progream at intermediate amateur strength.

### Quality metric
The quality of a model is measured as the number of games won in a playoff (alternating colors)
of 100 games against Leela 0.11.0 with parameters '--noponder --playouts=2000'.

### Quality target
The quality target is to win 50% of the games.

### Quality Progression
Informally, we have observed that quality should improve roughly linearly with time.  We observed
roughly 0.5% improvement in quality per hour of runtime. An example of approximately how we've seen
quality progress over time:

    Approx. Hours to Quality (16 CPU & 1 P100)
     1h           x%
     2h           x%
     4h           x%
     8h           x%

Note that quality does not necessarily monotonically increase.

### Target evaluation frequency
Target evaluation only needs to be performed for models which pass model evaluation.
