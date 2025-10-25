import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from aim_trainer_env import AimTrainerEnv


def make_env(render_mode=None, seed=42, max_steps=5000):
    """Create and wrap the AimTrainer environment"""
    env = AimTrainerEnv(
        render_mode=render_mode,
        seed=seed,
        max_steps=max_steps
    )
    env = Monitor(env)
    return env

def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Aim Trainer")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--logdir", type=str, default="./tf_logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--modeldir", type=str, default="./models",
                        help="Directory to save models")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--reward_mode", type=str, default="accuracy",
                        choices=["survival", "accuracy"])

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    print("Starting Training")
    print(f"Training Steps: {args.timesteps:,}")
    print(f"Max Episode Steps: {args.max_steps}")
    print(f"Seed: {args.seed}")

    env = make_env(
        seed=args.seed,
        max_steps=args.max_steps
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,

        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True
    )

    save_name = f"ppo_aim_trainer_{args.reward_mode}"
    save_path = os.path.join(args.modeldir, save_name)
    model.save(save_path)

    print("Training completed!")
    print(f"Model saved to: {save_path}")

    print("Quick test:")
    obs, info = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if done or truncated:
            break

    print(f"Test Results - Steps: {steps}, Total Reward: {total_reward:.2f}, Score: {info['score']}")

    env.close()


if __name__ == "__main__":
    main()