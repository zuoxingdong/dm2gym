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
One liner to create the environment:
```python
import gym
env = gym.make('dm2gym:CheetahRun-v0')
```

More examples to specify the environment:
```python
env = gym.make('dm2gym:FishSwim-v0', environment_kwargs={'flat_observation': True})
env = gym.make('dm2gym:HopperHop-v0', visualize_reward=True)
```

# What's new
- 2019-10-18 (v0.2.0)
    - Sync to the latest API of DeepMind Control Suite
    - Support gym registration: create all `dm_control` environments via `gym.make`

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
