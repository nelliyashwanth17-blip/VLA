
import numpy as np
from stable_baselines3 import PPO
from vla_env import VLA2DEnv

model = PPO.load("models/ppo_vla_tabletop", device="cpu")
env = VLA2DEnv()

print("PPO action diagnostic")
print("---------------------")

for episode in range(5):

    obs, info = env.reset(seed=1000 + episode)

    print(f"\nEpisode {episode + 1}")

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        before = info["distance"]

        obs, reward, terminated, truncated, info = env.step(action)

        after = info["distance"]

        print(
            f"Action: {action}, "
            f"Distance: {before} -> {after}, "
            f"Reward: {reward:.3f}"
        )

        done = terminated or truncated

env.close()
