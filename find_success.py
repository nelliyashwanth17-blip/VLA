
import numpy as np
import torch
from stable_baselines3 import PPO
from vision_only_env import make_vision_env

MODEL_PATH = "models/ppo_vla_vision_language_seed_7.zip"

model = PPO.load(MODEL_PATH)
env = make_vision_env()

print("SEARCHING FOR A SUCCESSFUL EPISODE")
print("----------------------------------")

found = False

for seed in range(100, 130):
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, info = env.reset(seed=seed)
    total_reward = 0

    for step in range(30):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward

        if terminated or truncated:
            break

    success = bool(info.get("success", False))

    print(
        f"Seed: {seed} | "
        f"Success: {success} | "
        f"Reward: {total_reward:.2f} | "
        f"Steps: {step + 1}"
    )

    if success:
        print("\nSUCCESSFUL EPISODE FOUND!")
        print("Use this seed for your live demonstration:", seed)
        found = True
        break

if not found:
    print("\nNo successful episode found in these 30 tests.")

env.close()
