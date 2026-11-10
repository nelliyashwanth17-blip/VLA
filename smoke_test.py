from gymnasium.utils.env_checker import check_env
from vla_env import VLA2DEnv

env = VLA2DEnv()
check_env(env)
obs, info = env.reset(seed=7)
print("Instruction:", info["instruction_text"])
print("Image shape:", obs["image"].shape)
print("Instruction vector:", obs["instruction"])
print("State vector:", obs["state"])

for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("reward:", round(reward, 3), "success:", info["success"])
    if terminated or truncated:
        break

env.close()
print("Environment smoke test passed.")
