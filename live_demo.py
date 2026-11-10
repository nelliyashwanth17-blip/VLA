
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from vision_only_env import make_vision_env

MODEL_PATH = "models/ppo_vla_vision_language_seed_7.zip"

# Set random seeds for the successful test episode
torch.manual_seed(100)
np.random.seed(100)

# Load trained model and environment
env = make_vision_env()
model = PPO.load(MODEL_PATH)

obs, info = env.reset(seed=100)

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

success = False
total_reward = 0.0
steps_taken = 0

print("\nVISION-LANGUAGE PPO LIVE DEMONSTRATION")
print("--------------------------------------")
print("Instruction: Move the red block next to the blue block")

for step in range(30):
    image = env.unwrapped.render()

    ax.clear()
    ax.imshow(image)
    ax.set_title(f"Vision-Language PPO | Step {step + 1}")
    ax.axis("off")
    plt.pause(0.5)

    action, _ = model.predict(obs, deterministic=False)
    action = int(action)

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    }

    print(f"Step {step + 1}: Action = {action_names[action]}")

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    steps_taken = step + 1

    if terminated or truncated:
        success = bool(info.get("success", False))
        break

image = env.unwrapped.render()
ax.clear()
ax.imshow(image)
ax.set_title("GOAL ACHIEVED!" if success else "EPISODE FINISHED")
ax.axis("off")

plt.pause(2)
plt.ioff()
plt.show()

print("\nDEMONSTRATION RESULTS")
print("---------------------")
print("Success:", success)
print("Total reward:", round(total_reward, 3))
print("Steps:", steps_taken)

env.close()
