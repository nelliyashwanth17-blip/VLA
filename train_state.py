
import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from vla_env import VLA2DEnv


class StateOnlyEnv(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "instruction": env.observation_space["instruction"],
            "state": env.observation_space["state"],
        })

    def observation(self, obs):
        return {
            "instruction": obs["instruction"],
            "state": obs["state"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)

    env = Monitor(StateOnlyEnv(VLA2DEnv()))

    model = PPO(
        "MultiInputPolicy",
        env,
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
        device="cpu",
    )

    print(f"Starting state-only PPO training for {args.timesteps:,} steps.")

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False
    )

    model.save("models/ppo_vla_state_only")
    env.close()

    print("State-only model saved successfully.")


if __name__ == "__main__":
    main()
