
from stable_baselines3 import PPO

models = {
    "State-only PPO": "models/ppo_vla_state_only",
    "MLP PPO": "models/ppo_vla_mlp",
    "Multimodal PPO": "models/ppo_vla_tabletop",
}

for name, path in models.items():
    model = PPO.load(path, device="cpu")

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Observation space:", model.observation_space)
    print("Policy architecture:")
    print(model.policy)

    print("\nLearning parameters:")
    print("Learning rate:", model.learning_rate)
    print("Network architecture:", model.policy.net_arch)
    print("Number of parameters:", sum(
        p.numel() for p in model.policy.parameters()
    ))
