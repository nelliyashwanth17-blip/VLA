
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from vla_env import VLA2DEnv


class MLPObservationEnv(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

    def observation(self, obs):

        return np.concatenate([
            obs["instruction"],
            obs["state"]
        ]).astype(np.float32)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)

    env = Monitor(MLPObservationEnv(VLA2DEnv()))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64]
            }
        },
        verbose=1,
        seed=args.seed,
        device="cpu"
    )

    print(
        f"Starting MLP PPO training for "
        f"{args.timesteps:,} steps."
    )

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False
    )

    model.save("models/ppo_vla_mlp")

    env.close()

    print("MLP PPO model saved successfully.")


if __name__ == "__main__":
    main()
