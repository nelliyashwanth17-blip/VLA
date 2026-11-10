
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from vision_only_env import make_vision_env


SEEDS = [42, 123, 2026, 7, 99]
MODES = ["deterministic", "stochastic"]
EPISODES = 100


def evaluate_model(seed, mode):
    model_path = f"models/ppo_vla_vision_language_seed_{seed}"

    model = PPO.load(model_path, device="cpu")
    env = make_vision_env()

    deterministic = mode == "deterministic"

    if not deterministic:
        set_random_seed(10000 + seed)

    results = []

    for episode in range(EPISODES):
        obs, info = env.reset(seed=1000 + episode)

        total_reward = 0.0
        steps = 0
        latencies = []
        success = False

        while True:
            import time
            start = time.perf_counter()

            action, _ = model.predict(
                obs,
                deterministic=deterministic
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
            "seed": seed,
            "mode": mode,
            "episode": episode + 1,
            "success": int(success),
            "reward": total_reward,
            "episode_length": steps,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95))
        })

    env.close()
    return results


def main():
    os.makedirs("results", exist_ok=True)

    all_results = []

    for seed in SEEDS:
        for mode in MODES:
            print("\nEvaluating seed:", seed, "| Mode:", mode)

            results = evaluate_model(seed, mode)
            all_results.extend(results)

            success_rate = np.mean(
                [r["success"] for r in results]
            ) * 100

            mean_reward = np.mean(
                [r["reward"] for r in results]
            )

            print("Success rate:", round(success_rate, 2), "%")
            print("Mean reward:", round(mean_reward, 3))

    episode_file = "results/vision_multi_seed_episodes.csv"

    with open(episode_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    summary = []

    for seed in SEEDS:
        for mode in MODES:
            rows = [
                r for r in all_results
                if r["seed"] == seed and r["mode"] == mode
            ]

            summary.append({
                "seed": seed,
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
                )
            })

    summary_file = "results/vision_multi_seed_summary.csv"

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n========== EVALUATION COMPLETED ==========")
    print("Episode results:", episode_file)
    print("Summary results:", summary_file)


if __name__ == "__main__":
    main()
