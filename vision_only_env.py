
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from vla_env import VLA2DEnv


class VisionOnlyEnv(gym.ObservationWrapper):
    """
    Vision + instruction environment.

    The agent receives:
    - RGB image
    - Language instruction encoding

    The agent does NOT receive exact block coordinates.
    """

    def __init__(self, env=None):
        if env is None:
            env = VLA2DEnv()

        super().__init__(env)

        image_shape = env.observation_space["image"].shape
        height, width, channels = image_shape

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(channels, height, width),
                dtype=np.uint8,
            ),
            "instruction": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        })

    def observation(self, obs):
        # Convert image from HWC to CHW format for CNN processing.
        image = np.transpose(obs["image"], (2, 0, 1)).copy()

        return {
            "image": image,
            "instruction": obs["instruction"].copy(),
        }


def make_vision_env():
    return VisionOnlyEnv(VLA2DEnv())
