
import pandas as pd

input_file = "results/vision_multi_seed_summary.csv"
output_file = "results/vision_language_final_statistics.csv"

df = pd.read_csv(input_file)

metrics = [
    "success_rate_percent",
    "mean_reward",
    "mean_episode_length",
    "mean_action_latency_ms",
    "mean_p95_latency_ms",
]

summary = df.groupby("mode")[metrics].agg(["mean", "std"])

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(output_file, index=False)

print("\nFINAL FIVE-SEED STATISTICS")
print(summary.to_string(index=False))

print("\nSaved to:", output_file)
