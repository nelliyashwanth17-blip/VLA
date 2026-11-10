
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from vla_env import VLA2DEnv

model = PPO.load("models/ppo_vla_tabletop", device="cpu")

results = []

for run in range(5):

    set_random_seed(100 + run)

    env = VLA2DEnv()
    successes = []
    rewards = []
    lengths = []

    for episode in range(100):

        obs, info = env.reset(seed=1000 + episode)

        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=False
            )

            obs, reward, terminated, truncated, info = env.step(
                int(action)
            )

            total_reward += reward
            steps += 1
            success = bool(info["success"])
            done = terminated or truncated

        successes.append(int(success))
        rewards.append(total_reward)
        lengths.append(steps)

    env.close()

    success_rate = 100 * np.mean(successes)

    results.append(success_rate)

    print(
        f"Run {run + 1}: "
        f"Success={success_rate:.1f}%, "
        f"Reward={np.mean(rewards):.3f}, "
        f"Length={np.mean(lengths):.2f}"
    )

print("\nFINAL STOCHASTIC RESULTS")
print("------------------------")
print(f"Mean success rate: {np.mean(results):.2f}%")
print(f"Standard deviation: {np.std(results, ddof=1):.2f}%")
print(f"Minimum: {np.min(results):.2f}%")
print(f"Maximum: {np.max(results):.2f}%")
