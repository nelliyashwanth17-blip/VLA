import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from vla_env import VLA2DEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    env = Monitor(VLA2DEnv())
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
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    print(f"Starting CPU training for {args.timesteps:,} environment steps.")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save("models/ppo_vla_tabletop")
    env.close()
    print("Saved model to models/ppo_vla_tabletop.zip")


if __name__ == "__main__":
    main()
