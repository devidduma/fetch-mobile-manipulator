# fetch-mobile-manipulator
Off-policy learning in robotics simulation with Fetch mobile manipulator.

## Abstract

In robotics, physics-based simulations are crucial for training real-life robots. Simulations have seen adoption accelerated by the rapid growth in computational power over the last three decades (Liu & Negrut, 2021). Robots are very complicated systems, training them in the real world can be challenging, since execution and feedback is slow. Physics-based simulation allows sampling experience millions times faster than in the real world, making it possible to train very complicated robots. 

In this project, we apply Off-Policy Deep Reinforcement Learning methods to Fetch Mobile Manipulator, a 7-DoF arm with a two-fingered parallel gripper attached to it. We train the Fetch robot to solve tasks such as Reach, Push, Pick and Place or Slide.

## Supported algorithms

Supported algorithms are listed below:
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/pdf/2101.05982.pdf)
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495)

## Usage
We can either train each task individually, or use Jupyter Notebooks to run multiple tasks, one after the other.
- In [first_benchmark](./first_benchmark), we trained each task individually.
- In [second_benchmark](./second_benchmark), we propose using Jupyter Notebooks to train one task after the other for each environment.

### Training with run configurations
Training with run configurations, like we did in the first benchmark, can be achieved using *run configurations* in PyCharm or another IDE. We have uploaded the run configurations used in the first benchmark.

### Training with Jupyter Notebooks
Training with Jupyter Notebooks, like we propose in the second benchmark, can be achieved using the following steps.

For each environment, a Jupyter Notebook is available to train 9 Deep Reinforcement Learning algorithms. To run the benchmark, go through the following steps:
1. Create a virtual environment and install all dependencies in it from [requirements.txt](./requirements.txt).
2. Execute all tasks in one notebook, to benchmark 9 DRL algorithms in that specific environment. To best use computational resources, we suggest executing 3 notebooks at a time. Each notebook takes approximately the same time to execute as other notebooks.
3. When execution is complete, multiple details are saved in the notebooks. This includes speed of training, running times for training and testing, best scores etc. Pretrained agents and logs are then saved in [benchmark/log](./benchmark/log) folder and can be monitored with Tensorboard to generate graph plots.

```bash
$ tensorboard --logdir log
```

## Results from first benchmark
All results from the first benchmark in our experiments are saved in the [first_benchmark](./first_benchmark) folder.
 - [first_benchmark/log](./first_benchmark/log) folder: contains all pretrained agents and logs, which can be plotted with Tensorboard.
 - [first_benchmark/plots](./first_benchmark/plots) folder: graph plots generated with Tensorboard.
 - [first_benchmark/outputs](./first_benchmark/outputs) folder: contains command line outputs from the beginning of training until the end of training for each task.

## Video demonstrations
For each task, we have generated video demonstrations of our pretrained agents and saved the videos in the [demonstrations](./demonstrations) folder.

To generate new videos, simply run the script [demonstrations/demonstrations.py](./demonstrations/demonstrations.py). This will generate new videos for all tasks in all environments in batch.
