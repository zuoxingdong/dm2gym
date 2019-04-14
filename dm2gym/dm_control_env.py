from collections import OrderedDict

import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from dm_control.rl.specs import ArraySpec
from dm_control.rl.specs import BoundedArraySpec


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, BoundedArraySpec):
        space = spaces.Box(low=dm_control_space.minimum, 
                           high=dm_control_space.maximum, 
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, ArraySpec) and not isinstance(dm_control_space, BoundedArraySpec):
        space = spaces.Box(low=-float('inf'), 
                           high=float('inf'), 
                           shape=dm_control_space.shape, 
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, OrderedDict):
        space = spaces.Dict(OrderedDict([(key, convert_dm_control_to_gym_space(value)) 
                                         for key, value in dm_control_space.items()]))
        return space
    

class DMControlEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': int(np.round(1.0/self.env.control_timestep()))}

        self.observation_space = convert_dm_control_to_gym_space(env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(env.action_spec())
        max_episode_steps = None if env._step_limit == float('inf') else int(env._step_limit)
        self.spec = EnvSpec('DM-v0', max_episode_steps=max_episode_steps)
        self.viewer = None
    
    def seed(self, seed):
        return self.env.task.random.seed(seed)
    
    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        timestep = self.env.reset()
        return timestep.observation
    
    def render(self, mode='human', **kwargs):
        img = self.env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=1024)
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
