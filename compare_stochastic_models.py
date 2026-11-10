
import csv
import os
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from vla_env import VLA2DEnv


# --------------------------------------------------
# Observation wrappers
# --------------------------------------------------

class StateOnlyEnv(gym.ObservationWrapper):
    """Keep the instruction and state, but remove the image."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "instruction": env.observation_space["instruction"],
            "state": env.observation_space["state"],
        })

    def observation(self, observation):
        return {
            "instruction": observation["instruction"],
            "state": observation["state"],
        }


class MLPObservationEnv(gym.ObservationWrapper):
    """Combine instruction and state into one six-value vector."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.concatenate([
            observation["instruction"],
            observation["state"],
        ]).astype(np.float32)


# --------------------------------------------------
# Success detection
# --------------------------------------------------

def check_success(original_observation, info):
    if "is_success" in info:
        return bool(info["is_success"])

    if "success" in info:
        return bool(info["success"])

    instruction = np.asarray(
        original_observation["instruction"]
    )
    state = np.asarray(original_observation["state"])

    red_x, red_y, blue_x, blue_y = np.rint(
        state * 5
    ).astype(int)

    red_position = (red_x, red_y)
    blue_position = (blue_x, blue_y)

    if instruction[0] > instruction[1]:
        selected = red_position
        other = blue_position
    else:
        selected = blue_position
        other = red_position

    distance = (
        abs(selected[0] - other[0])
        + abs(selected[1] - other[1])
    )

    return distance == 1


# --------------------------------------------------
# Models
# --------------------------------------------------

MODELS = {
    "Multimodal PPO": {
        "path": "models/ppo_vla_tabletop",
        "wrapper": "multimodal",
    },
    "State-only PPO": {
        "path": "models/ppo_vla_state_only",
        "wrapper": "state",
    },
    "MLP PPO": {
        "path": "models/ppo_vla_mlp",
        "wrapper": "mlp",
    },
}


def create_environment(wrapper_type):
    env = VLA2DEnv()

    if wrapper_type == "state":
        env = StateOnlyEnv(env)
    elif wrapper_type == "mlp":
        env = MLPObservationEnv(env)

    return env


# --------------------------------------------------
# Stochastic evaluation
# --------------------------------------------------

def evaluate_model(model_name, model_config, sampling_seed,
                   episodes=100):

    set_random_seed(sampling_seed)

    env = create_environment(model_config["wrapper"])
    base_env = env.unwrapped

    model = PPO.load(model_config["path"], device="cpu")

    episode_rows = []

    for episode in range(episodes):

        observation, info = env.reset(seed=1000 + episode)

        total_reward = 0.0
        episode_length = 0
        successful = False
        action_latencies = []

        while True:

            start_time = time.perf_counter()

            action, _ = model.predict(
                observation,
                deterministic=False,
            )

            action_latencies.append(
                (time.perf_counter() - start_time) * 1000
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)
            episode_length += 1

            original_observation = base_env._get_obs()

            if check_success(original_observation, info):
                successful = True

            if terminated or truncated:
                break

        episode_rows.append({
            "model": model_name,
            "sampling_seed": sampling_seed,
            "episode": episode + 1,
            "success": int(successful),
            "reward": total_reward,
            "episode_length": episode_length,
            "mean_action_latency_ms": float(
                np.mean(action_latencies)
            ),
            "p95_action_latency_ms": float(
                np.percentile(action_latencies, 95)
            ),
        })

    env.close()
    return episode_rows


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    os.makedirs("results", exist_ok=True)

    sampling_seeds = [101, 102, 103, 104, 105]
    all_episode_results = []

    for model_name, model_config in MODELS.items():

        print(f"\nEvaluating: {model_name}")

        for sampling_seed in sampling_seeds:

            print(f"  Sampling seed: {sampling_seed}")

            rows = evaluate_model(
                model_name,
                model_config,
                sampling_seed,
                episodes=100,
            )

            all_episode_results.extend(rows)

            success_rate = np.mean(
                [row["success"] for row in rows]
            ) * 100

            print(f"    Success rate: {success_rate:.2f}%")

    episode_file = (
        "results/stochastic_architecture_episodes.csv"
    )

    with open(episode_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_episode_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_episode_results)

    # Summarize each model across its five sampling runs.
    summary_rows = []

    for model_name in MODELS:

        model_rows = [
            row for row in all_episode_results
            if row["model"] == model_name
        ]

        run_success_rates = []

        for seed in sampling_seeds:
            seed_rows = [
                row for row in model_rows
                if row["sampling_seed"] == seed
            ]

            run_success_rates.append(
                np.mean([row["success"] for row in seed_rows])
                * 100
            )

        summary_rows.append({
            "model": model_name,
            "mean_success_rate_percent": np.mean(
                run_success_rates
            ),
            "std_success_rate_percent": np.std(
                run_success_rates, ddof=1
            ),
            "min_success_rate_percent": np.min(
                run_success_rates
            ),
            "max_success_rate_percent": np.max(
                run_success_rates
            ),
            "mean_reward": np.mean(
                [row["reward"] for row in model_rows]
            ),
            "mean_episode_length": np.mean(
                [row["episode_length"] for row in model_rows]
            ),
            "mean_action_latency_ms": np.mean(
                [row["mean_action_latency_ms"] for row in model_rows]
            ),
            "mean_p95_action_latency_ms": np.mean(
                [row["p95_action_latency_ms"] for row in model_rows]
            ),
        })

    summary_file = (
        "results/stochastic_architecture_comparison.csv"
    )

    with open(summary_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== STOCHASTIC COMPARISON ==========")

    for row in summary_rows:
        print(f"\n{row['model']}")
        print(
            f"Success: {row['mean_success_rate_percent']:.2f}% "
            f"+/- {row['std_success_rate_percent']:.2f}%"
        )
        print(f"Reward: {row['mean_reward']:.3f}")
        print(
            f"Mean episode length: "
            f"{row['mean_episode_length']:.2f}"
        )
        print(
            f"Mean action latency: "
            f"{row['mean_action_latency_ms']:.4f} ms"
        )

    print("\nSaved episode results:", episode_file)
    print("Saved summary results:", summary_file)


if __name__ == "__main__":
    main()
