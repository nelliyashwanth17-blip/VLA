
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def read_long_format(filename, policy_name):
    """Read a policy's metrics from a long-format CSV."""
    df = pd.read_csv(f"{RESULTS}/{filename}")

    selected = df[df["policy"] == policy_name]

    return {
        row["metric"]: float(row["value"])
        for _, row in selected.iterrows()
    }


def main():

    # --------------------------------------------------
    # 1. DETERMINISTIC RESULTS
    # --------------------------------------------------

    baseline = read_long_format(
        "deterministic_multimodal_final.csv",
        "Rule-based baseline"
    )

    multimodal = read_long_format(
        "deterministic_multimodal_final.csv",
        "PPO policy"
    )

    state_only = read_long_format(
        "deterministic_state_final.csv",
        "State-only PPO"
    )

    mlp = read_long_format(
        "deterministic_mlp_final.csv",
        "MLP PPO"
    )

    vision_stats = pd.read_csv(
        f"{RESULTS}/vision_language_final_statistics.csv"
    )

    vision_det = vision_stats[
        vision_stats["mode"] == "deterministic"
    ].iloc[0]

    vision_sto = vision_stats[
        vision_stats["mode"] == "stochastic"
    ].iloc[0]

    deterministic_rows = [
        {
            "model": "Rule-based baseline",
            "success_rate_percent": baseline["success_rate_percent"],
            "mean_reward": baseline["mean_reward"],
            "mean_episode_length": baseline["mean_episode_length"],
            "mean_action_latency_ms": baseline["mean_action_latency_ms"],
            "evidence": "Previously recorded 100-episode evaluation"
        },
        {
            "model": "Multimodal PPO",
            "success_rate_percent": multimodal["success_rate_percent"],
            "mean_reward": multimodal["mean_reward"],
            "mean_episode_length": multimodal["mean_episode_length"],
            "mean_action_latency_ms": multimodal["mean_action_latency_ms"],
            "evidence": "Previously recorded 100-episode evaluation"
        },
        {
            "model": "State-only PPO",
            "success_rate_percent": state_only["success_rate_percent"],
            "mean_reward": state_only["mean_reward"],
            "mean_episode_length": state_only["mean_episode_length"],
            "mean_action_latency_ms": state_only["mean_action_latency_ms"],
            "evidence": "Previously recorded 100-episode evaluation"
        },
        {
            "model": "MLP PPO",
            "success_rate_percent": mlp["success_rate_percent"],
            "mean_reward": mlp["mean_reward"],
            "mean_episode_length": mlp["mean_episode_length"],
            "mean_action_latency_ms": mlp["mean_action_latency_ms"],
            "evidence": "Previously recorded 100-episode evaluation"
        },
        {
            "model": "Vision-Language PPO",
            "success_rate_percent": vision_det["success_rate_percent_mean"],
            "success_rate_std": vision_det["success_rate_percent_std"],
            "mean_reward": vision_det["mean_reward_mean"],
            "mean_reward_std": vision_det["mean_reward_std"],
            "mean_episode_length": vision_det["mean_episode_length_mean"],
            "mean_action_latency_ms": vision_det["mean_action_latency_ms_mean"],
            "evidence": "Five independent training seeds; 100 episodes per seed"
        }
    ]

    deterministic_df = pd.DataFrame(deterministic_rows)

    deterministic_df.to_csv(
        f"{RESULTS}/final_deterministic_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # 2. STOCHASTIC RESULTS
    # --------------------------------------------------

    old_stochastic = pd.read_csv(
        f"{RESULTS}/stochastic_architecture_comparison.csv"
    )

    stochastic_rows = []

    for _, row in old_stochastic.iterrows():
        stochastic_rows.append({
            "model": row["model"],
            "success_rate_percent": row["mean_success_rate_percent"],
            "success_rate_std": row["std_success_rate_percent"],
            "mean_reward": row["mean_reward"],
            "mean_episode_length": row["mean_episode_length"],
            "mean_action_latency_ms": row["mean_action_latency_ms"],
            "evidence": "Previously recorded stochastic architecture comparison; run count requires verification"
        })

    stochastic_rows.append({
        "model": "Vision-Language PPO",
        "success_rate_percent": vision_sto["success_rate_percent_mean"],
        "success_rate_std": vision_sto["success_rate_percent_std"],
        "mean_reward": vision_sto["mean_reward_mean"],
        "mean_episode_length": vision_sto["mean_episode_length_mean"],
        "mean_action_latency_ms": vision_sto["mean_action_latency_ms_mean"],
        "evidence": "Five independent training seeds; 100 episodes per seed"
    })

    stochastic_df = pd.DataFrame(stochastic_rows)

    stochastic_df.to_csv(
        f"{RESULTS}/final_stochastic_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # 3. MASTER COMPARISON CSV
    # --------------------------------------------------

    deterministic_master = deterministic_df.copy()
    deterministic_master["mode"] = "deterministic"

    stochastic_master = stochastic_df.copy()
    stochastic_master["mode"] = "stochastic"

    master = pd.concat(
        [deterministic_master, stochastic_master],
        ignore_index=True
    )

    master.to_csv(
        f"{RESULTS}/final_consolidated_results.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. GRAPH 1: DETERMINISTIC SUCCESS
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        deterministic_df["model"],
        deterministic_df["success_rate_percent"]
    )

    plt.ylabel("Success Rate (%)")
    plt.xlabel("Model")
    plt.title("Deterministic Success Rate Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 110)
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS}/final_deterministic_success.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # 5. GRAPH 2: STOCHASTIC SUCCESS
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        stochastic_df["model"],
        stochastic_df["success_rate_percent"]
    )

    plt.ylabel("Success Rate (%)")
    plt.xlabel("Model")
    plt.title("Stochastic Success Rate Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS}/final_stochastic_success.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # 6. GRAPH 3: MEAN REWARD
    # --------------------------------------------------

    reward_df = pd.concat([
        deterministic_df[["model", "mean_reward"]].assign(
            mode="Deterministic"
        ),
        stochastic_df[["model", "mean_reward"]].assign(
            mode="Stochastic"
        )
    ])

    reward_pivot = reward_df.pivot(
        index="model",
        columns="mode",
        values="mean_reward"
    )

    ax = reward_pivot.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Model")
    ax.set_title("Mean Reward: Deterministic vs Stochastic")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS}/final_mean_reward_comparison.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # 7. GRAPH 4: INFERENCE LATENCY
    # --------------------------------------------------

    latency_df = pd.concat([
        deterministic_df[["model", "mean_action_latency_ms"]].assign(
            mode="Deterministic"
        ),
        stochastic_df[["model", "mean_action_latency_ms"]].assign(
            mode="Stochastic"
        )
    ])

    latency_pivot = latency_df.pivot(
        index="model",
        columns="mode",
        values="mean_action_latency_ms"
    )

    ax = latency_pivot.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_ylabel("Mean Action Latency (ms)")
    ax.set_xlabel("Model")
    ax.set_title("Inference Latency Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS}/final_latency_comparison.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # 8. COMPLETION
    # --------------------------------------------------

    print("\n======================================")
    print("FINAL RESEARCH RESULTS GENERATED")
    print("======================================")

    print("\nCSV FILES:")
    print("final_deterministic_comparison.csv")
    print("final_stochastic_comparison.csv")
    print("final_consolidated_results.csv")

    print("\nGRAPHS:")
    print("final_deterministic_success.png")
    print("final_stochastic_success.png")
    print("final_mean_reward_comparison.png")
    print("final_latency_comparison.png")

    print("\nVISION-LANGUAGE FIVE-SEED SUMMARY:")
    print(
        vision_stats[
            [
                "mode",
                "success_rate_percent_mean",
                "success_rate_percent_std",
                "mean_reward_mean",
                "mean_episode_length_mean"
            ]
        ].to_string(index=False)
    )

    print("\nAll files saved in the results folder.")


if __name__ == "__main__":
    main()
