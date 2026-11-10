
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from vision_only_env import make_vision_env


def train_model(seed, timesteps):
    set_random_seed(seed)

    env = make_vision_env()

    print("\n" + "=" * 55)
    print("VISION-LANGUAGE PPO TRAINING")
    print("=" * 55)
    print("Training seed:", seed)
    print("Timesteps:", timesteps)
    print("Device: CPU")
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
        seed=seed,
        device="cpu",
    )

    model.learn(total_timesteps=timesteps)

    save_path = f"models/ppo_vla_vision_language_seed_{seed}"
    model.save(save_path)

    print("\nTraining completed.")
    print("Seed:", seed)
    print("Actual timesteps:", model.num_timesteps)
    print("Saved model:", save_path + ".zip")

    env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=12000
    )

    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    seeds = [42, 123, 2026, 7, 99]

    for seed in seeds:
        train_model(seed, args.timesteps)

    print("\n" + "=" * 55)
    print("ALL FIVE TRAINING RUNS COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
