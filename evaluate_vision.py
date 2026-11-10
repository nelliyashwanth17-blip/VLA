
import argparse
import csv
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from vision_only_env import make_vision_env


def evaluate(model, env, episodes, deterministic, mode):
    results = []

    for episode in range(episodes):
        obs, info = env.reset(seed=1000 + episode)

        total_reward = 0.0
        steps = 0
        latencies = []
        success = False

        while True:
            start = time.perf_counter()

            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            latencies.append(
                (time.perf_counter() - start) * 1000
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            success = bool(info.get("success", False))

            if terminated or truncated:
                break

        results.append({
            "mode": mode,
            "episode": episode + 1,
            "success": int(success),
            "reward": total_reward,
            "episode_length": steps,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    model_path = "models/ppo_vla_vision_language"

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            "Vision-Language PPO model not found."
        )

    os.makedirs("results", exist_ok=True)

    model = PPO.load(model_path, device="cpu")
    env = make_vision_env()

    all_results = []

    for mode, deterministic in [
        ("deterministic", True),
        ("stochastic", False),
    ]:
        if not deterministic:
            set_random_seed(101)

        print(f"\nEvaluating Vision-Language PPO: {mode}")

        results = evaluate(
            model,
            env,
            args.episodes,
            deterministic,
            mode,
        )

        all_results.extend(results)

        success_rate = np.mean(
            [r["success"] for r in results]
        ) * 100

        mean_reward = np.mean(
            [r["reward"] for r in results]
        )

        mean_length = np.mean(
            [r["episode_length"] for r in results]
        )

        mean_latency = np.mean(
            [r["mean_latency_ms"] for r in results]
        )

        print(f"Success rate: {success_rate:.2f}%")
        print(f"Mean reward: {mean_reward:.3f}")
        print(f"Mean episode length: {mean_length:.2f}")
        print(f"Mean action latency: {mean_latency:.4f} ms")

    episode_file = (
        "results/vision_language_evaluation.csv"
    )

    with open(episode_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_results)

    summary = []

    for mode in ["deterministic", "stochastic"]:
        rows = [r for r in all_results if r["mode"] == mode]

        summary.append({
            "model": "Vision-Language PPO",
            "mode": mode,
            "episodes": len(rows),
            "success_rate_percent": np.mean(
                [r["success"] for r in rows]
            ) * 100,
            "mean_reward": np.mean(
                [r["reward"] for r in rows]
            ),
            "mean_episode_length": np.mean(
                [r["episode_length"] for r in rows]
            ),
            "mean_action_latency_ms": np.mean(
                [r["mean_latency_ms"] for r in rows]
            ),
            "mean_p95_latency_ms": np.mean(
                [r["p95_latency_ms"] for r in rows]
            ),
        })

    summary_file = (
        "results/vision_language_summary.csv"
    )

    with open(summary_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n========== EVALUATION COMPLETED ==========")
    print("Episode results:", episode_file)
    print("Summary results:", summary_file)

    env.close()


if __name__ == "__main__":
    main()
