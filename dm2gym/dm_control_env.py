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

    def __init__(self, env, *, render_window_mode='gym'):
        """Constructor

        Args:
            env (dm_control.rl.control.Environment) The dm_control environment
                to wrap
            render_window_mode (str): Which render window mode to use. Options
                are;
                - 'gym' (default): Use
                    `gym.envs.classic_control.rendering.SimpleImageViewer`,
                    which is backed by pyglet
                - 'opencv': Use an OpenCV rendering window mode
        """

        # Verify render window mode
        assert render_window_mode in ['gym', 'opencv'],\
            "Invalid value for render_window_mode: {}".format(
                render_window_mode
            )

        self.env = env
        self.render_window_mode = render_window_mode
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

        if 'camera_id' not in kwargs:
            # Tracking camera
            kwargs['camera_id'] = 0

        img = self.env.physics.render(**kwargs)
        
        if mode == 'rgb_array':

            return img

        elif mode == "human":

            if self.render_window_mode == 'gym':

                # Use a gym-backed render window
                from gym.envs.classic_control import rendering

                # Construct viewer
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1024)

                self.viewer.imshow(img)
                
                return self.viewer.isopen

            elif self.render_window_mode == 'opencv':

                # Use an opencv-backed render window
                import cv2

                # Construct viewer, saving the window ID string to self.viewer
                if self.viewer is None:
                    self.viewer = self.env.__str__()
                    cv2.namedWindow(self.viewer, cv2.WINDOW_AUTOSIZE)

                # Convert to BGR and show
                cv2.imshow(self.viewer, img[:, :, [2, 1, 0]])

                # Listen for escape key, then exit if pressed
                if cv2.waitKey(1) in [27]:
                    exit()

                return True


            else:

                raise ValueError(
                    "Invalid value for render_window_mode: {}".format(
                        self.render_window_mode
                    )
                )

        else:

            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
