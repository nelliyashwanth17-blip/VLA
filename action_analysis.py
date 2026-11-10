
import numpy as np
from collections import Counter

from stable_baselines3 import PPO
from vla_env import VLA2DEnv

model = PPO.load("models/ppo_vla_tabletop", device="cpu")
env = VLA2DEnv()

action_names = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

counts = Counter()
probability_sum = np.zeros(4)
samples = 0

for episode in range(100):

    obs, info = env.reset(seed=2000 + episode)

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs_tensor, _ = model.policy.obs_to_tensor(obs)

        distribution = model.policy.get_distribution(obs_tensor)
        probabilities = distribution.distribution.probs

        probability_sum += probabilities.detach().cpu().numpy()[0]

        counts[action] += 1
        samples += 1

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

print("\nPPO ACTION DISTRIBUTION")
print("-----------------------")

for action in range(4):
    print(
        f"{action_names[action]}: "
        f"{counts[action]} actions "
        f"({counts[action] / samples * 100:.2f}%)"
    )

print("\nAVERAGE ACTION PROBABILITIES")
print("----------------------------")

average_probabilities = probability_sum / samples

for action in range(4):
    print(
        f"{action_names[action]}: "
        f"{average_probabilities[action]:.4f}"
    )

env.close()
