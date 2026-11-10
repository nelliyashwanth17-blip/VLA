
import numpy as np
from collections import Counter

from stable_baselines3 import PPO
from vla_env import VLA2DEnv

model = PPO.load("models/ppo_vla_tabletop", device="cpu")

action_names = ["UP", "DOWN", "LEFT", "RIGHT"]


def evaluate(deterministic, episodes=100):

    env = VLA2DEnv()
    successes = []
    action_counts = Counter()

    for episode in range(episodes):

        obs, _ = env.reset(seed=1000 + episode)
        done = False
        success = False

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=deterministic
            )

            action = int(action)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            success = bool(info["success"])
            done = terminated or truncated

        successes.append(int(success))

    env.close()

    return (
        100 * np.mean(successes),
        action_counts
    )


for name, deterministic in [
    ("Deterministic", True),
    ("Stochastic", False)
]:

    success_rate, counts = evaluate(deterministic)

    print(f"\n{name} policy")
    print(f"Success rate: {success_rate:.2f}%")

    total = sum(counts.values())

    for action in range(4):
        percentage = 100 * counts[action] / total

        print(
            f"{action_names[action]}: "
            f"{percentage:.2f}%"
        )
