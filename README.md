# Tennis

Unity's Tennis Environrment is an environment in which an agent must 
control two tennis rackets with the goal of hitting a ball back and forth between them. The agent is rewarded when the ball passes over the net to the other racket and penalized when the ball falls to the ground.

The agent interacts with the environment via the following:
 - It is fed two sets of observations (one for each racket). Each observation is a vector of 8 elements
 - For each racket, the agent must provide an action vector representing 2 continuous actions

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

*Sample gameplay taken from Udacity Repository: https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet* 

This repository trains an agent to attain an average score (over 100 episodes) of at least 0.5. It trains the agent using [Multi Agent Deep Deterministic Policy Gradients](https://arxiv.org/abs/1706.02275).

## Prerequisites

- Anaconda
- Python 3.6
- A `conda` environment created as follows

  - Linux or Mac:
  ```
  conda create --name drlnd python=3.6
  source activate drlnd 
  ```

  - Windows
  ```
  conda create --name drlnd python=3.6 
  activate drlnd
  ```

- Required dependencies

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

## Getting Started

1. `git clone https://github.com/JoshVarty/Tennis.git`

2. `cd Tennis`

3. Download Unity Tennis Environment
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
4. Unzip to git directory

5. `jupyter notebook`

6. You can train your own agent via [`main.ipynb`](https://github.com/JoshVarty/Tennis/blob/master/main.ipynb) or watch a single episode of the pre-trained network via [`Visualization.ipynb`](https://github.com/JoshVarty/Tennis/blob/master/Visualization.ipynb)


## Results

After approximately two hours of training (3,900 episodes) on my local machine, the agent achieved a score greater than `0.5`. Its progress is shown below:

![https://i.imgur.com/Qyb4dtJ.png](https://i.imgur.com/Qyb4dtJ.png)

A visualization of the agent in action:

![https://i.imgur.com/Yk9IJux.gif](https://i.imgur.com/Yk9IJux.gif)

## Notes
 - Only tested on Ubuntu 18.04
 - Details of the learning algorithm, architecture and hyperparameters can be found in `Report.md`