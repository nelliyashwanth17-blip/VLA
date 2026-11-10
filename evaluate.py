import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from vla_env import VLA2DEnv
from baseline import heuristic_action


def evaluate_policy(env, model=None, episodes=100, baseline=False):
    successes, rewards, lengths, latencies = [], [], [], []

    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        total_reward = 0.0
        steps = 0
        success = False
        done = False

        while not done:
            start = time.perf_counter()
            if baseline:
                action = heuristic_action(obs, env.grid_size)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            latencies.append((time.perf_counter() - start) * 1000.0)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            success = bool(info["success"])
            done = terminated or truncated

        successes.append(int(success))
        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "success_rate_percent": 100.0 * float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_action_latency_ms": float(np.mean(latencies)),
        "p95_action_latency_ms": float(np.percentile(latencies, 95)),
    }


def main():
    Path("results").mkdir(exist_ok=True)
    env = VLA2DEnv()
    model = PPO.load("models/ppo_vla_tabletop", device="cpu")

    print("Evaluating rule-based baseline...")
    baseline = evaluate_policy(env, baseline=True)
    print("Evaluating PPO policy...")
    ppo = evaluate_policy(env, model=model)

    rows = [("Rule-based baseline", baseline), ("PPO policy", ppo)]
    for name, result in rows:
        print(f"\n{name}")
        for key, value in result.items():
            print(f"  {key}: {value:.3f}")

    with open("results/evaluation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "metric", "value"])
        for name, result in rows:
            for metric, value in result.items():
                writer.writerow([name, metric, value])

    metrics = [
        ("success_rate_percent", "Success rate (%)"),
        ("mean_action_latency_ms", "Mean action latency (ms)"),
        ("mean_episode_length", "Mean episode length (steps)"),
        ("mean_reward", "Mean episode reward"),
    ]
    for key, title in metrics:
        plt.figure(figsize=(6, 4))
        plt.bar([name for name, _ in rows], [result[key] for _, result in rows])
        plt.title(title)
        plt.ylabel(title)
        plt.tight_layout()
        filename = key + ".png"
        plt.savefig(Path("results") / filename, dpi=150)
        plt.close()

    env.close()
    print("\nSaved CSV and comparison graphs in results/.")


if __name__ == "__main__":
    main()
