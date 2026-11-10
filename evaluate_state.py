
import csv
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from vla_env import VLA2DEnv
from baseline import heuristic_action


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


def evaluate(env, model=None, episodes=100, baseline=False):

    successes = []
    rewards = []
    lengths = []
    latencies = []

    for episode in range(episodes):

        obs, _ = env.reset(seed=1000 + episode)

        total_reward = 0.0
        steps = 0
        success = False
        done = False

        while not done:

            start = time.perf_counter()

            if baseline:
                action = heuristic_action(obs, env.unwrapped.grid_size)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            latencies.append(
                (time.perf_counter() - start) * 1000
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            success = bool(info["success"])
            done = terminated or truncated

        successes.append(int(success))
        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "success_rate_percent": 100 * np.mean(successes),
        "mean_reward": np.mean(rewards),
        "mean_episode_length": np.mean(lengths),
        "mean_action_latency_ms": np.mean(latencies),
        "p95_action_latency_ms": np.percentile(latencies, 95),
    }


def main():

    Path("results").mkdir(exist_ok=True)

    env = StateOnlyEnv(VLA2DEnv())
    model = PPO.load("models/ppo_vla_state_only", device="cpu")

    print("Evaluating rule-based baseline...")
    baseline = evaluate(env, baseline=True)

    print("Evaluating state-only PPO...")
    ppo = evaluate(env, model=model)

    rows = [
        ("Rule-based baseline", baseline),
        ("State-only PPO", ppo)
    ]

    for name, result in rows:
        print(f"\n{name}")
        for key, value in result.items():
            print(f"  {key}: {value:.3f}")

    with open(
        "results/state_only_evaluation.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["policy", "metric", "value"])

        for name, result in rows:
            for metric, value in result.items():
                writer.writerow([name, metric, value])

    metrics = [
        ("success_rate_percent", "Success rate (%)"),
        ("mean_reward", "Mean reward"),
        ("mean_episode_length", "Mean episode length"),
        ("mean_action_latency_ms", "Mean action latency (ms)")
    ]

    for key, title in metrics:

        plt.figure(figsize=(6, 4))

        plt.bar(
            [name for name, _ in rows],
            [result[key] for _, result in rows]
        )

        plt.title(title)
        plt.ylabel(title)
        plt.tight_layout()

        plt.savefig(
            Path("results") / f"state_only_{key}.png",
            dpi=150
        )

        plt.close()

    env.close()

    print("\nState-only evaluation results saved successfully.")


if __name__ == "__main__":
    main()
