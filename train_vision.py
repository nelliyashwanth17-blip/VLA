
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from vision_only_env import make_vision_env


def main():
    parser = argparse.ArgumentParser(
        description="Train Vision-Language PPO"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=12000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_random_seed(args.seed)

    os.makedirs("models", exist_ok=True)

    env = make_vision_env()

    print("Starting Vision-Language PPO training...")
    print("Device: CPU")
    print("Training timesteps:", args.timesteps)
    print("Random seed:", args.seed)
    print("Observation: Image + Instruction")
    print("Exact simulator state: Not provided")

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64],
            }
        },
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    model.learn(
        total_timesteps=args.timesteps
    )

    save_path = "models/ppo_vla_vision_language"

    model.save(save_path)

    print("\nVision-Language PPO training completed.")
    print("Actual training timesteps:", model.num_timesteps)
    print("Model saved to:", save_path + ".zip")

    env.close()


if __name__ == "__main__":
    main()
