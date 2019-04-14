# `dm2gym`: Convert DeepMind Control Suite to OpenAI gym environments.

This repository contains a lightweight wrapper to convert DeepMind Control Suite to OpenAI gym environments. 

# Installation
One can install directly from PyPI:
```
pip install dm2gym
```
The installation can also be done with:
```
git clone https://github.com/zuoxingdong/dm2gym.git
cd dm2gym
pip install -e .
```

# Getting started
Converting the environment from `dm_control` to `gym` can be as simple as:
```python
from dm_control import suite
from dm2gym import DMControlEnv

env = suite.load('cheetah', 'run')
env = DMControlEnv(env)

```

# What's new

- 2019-04-14 (v0.1.0)
    - Initial release

# Reference
Please use this bibtex if you want to cite this repository in your publications:
    
    @misc{dm2gym,
          author = {Zuo, Xingdong},
          title = {dm2gym: Convert DeepMind Control Suite to OpenAI gym environments.},
          year = {2019},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/zuoxingdong/dm2gym}},
        }
