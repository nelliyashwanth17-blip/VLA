
import numpy as np
import torch
from stable_baselines3 import PPO
from vla_env import VLA2DEnv
import gymnasium as gym
from gymnasium import spaces


class StateOnlyEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "instruction": env.observation_space["instruction"],
            "state": env.observation_space["state"],
        })

    def observation(self, obs):
        return {
            "instruction": obs["instruction"],
            "state": obs["state"],
        }


class MLPObservationEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.concatenate([
            obs["instruction"],
            obs["state"],
        ]).astype(np.float32)


state_model = PPO.load("models/ppo_vla_state_only", device="cpu")
mlp_model = PPO.load("models/ppo_vla_mlp", device="cpu")

state_env = StateOnlyEnv(VLA2DEnv())

print("Comparing action probabilities on 20 identical states\n")

same_actions = 0
same_probabilities = 0

for i in range(20):

    original_obs, _ = state_env.reset(seed=2000 + i)

    state_obs = {
        "instruction": original_obs["instruction"],
        "state": original_obs["state"],
    }

    mlp_obs = np.concatenate([
        original_obs["instruction"],
        original_obs["state"],
    ]).astype(np.float32)

    # Convert observations into PyTorch tensors.
    state_tensor, _ = state_model.policy.obs_to_tensor(state_obs)
    mlp_tensor, _ = mlp_model.policy.obs_to_tensor(mlp_obs)

    with torch.no_grad():
        state_distribution = state_model.policy.get_distribution(
            state_tensor
        )
        mlp_distribution = mlp_model.policy.get_distribution(
            mlp_tensor
        )

        state_probs = (
            state_distribution.distribution.probs
            .cpu().numpy().flatten()
        )

        mlp_probs = (
            mlp_distribution.distribution.probs
            .cpu().numpy().flatten()
        )

    state_action, _ = state_model.predict(
        state_obs, deterministic=True
    )

    mlp_action, _ = mlp_model.predict(
        mlp_obs, deterministic=True
    )

    if np.array_equal(state_action, mlp_action):
        same_actions += 1

    if np.allclose(state_probs, mlp_probs, atol=1e-6):
        same_probabilities += 1

    print(f"State {i + 1}:")
    print("  State-only:", np.round(state_probs, 4))
    print("  MLP:       ", np.round(mlp_probs, 4))
    print(
        "  Actions:",
        int(np.asarray(state_action).item()),
        int(np.asarray(mlp_action).item()),
    )

print("\n========== SUMMARY ==========")
print("Same deterministic actions:", same_actions, "/ 20")
print("Same probability distributions:", same_probabilities, "/ 20")

state_env.close()
