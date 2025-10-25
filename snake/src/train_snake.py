import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from snake_env import SnakeEnv


def make_env(render_mode=None, reward_mode="survival", seed=42, max_steps=5000):
    env = SnakeEnv(
        render_mode=render_mode,
        reward_mode=reward_mode,
        seed=seed,
        max_steps=max_steps
    )
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train RL agent to play Snake")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--reward_mode", type=str, default="survival",
                        choices=["survival", "length"],
                        help="Reward function to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--logdir", type=str, default="./tf_logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--modeldir", type=str, default="./models",
                        help="Directory to save models")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum steps per episode")

    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps to run for each environment per update")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epoch when optimizing the surrogate loss")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="Factor for trade-off of bias vs variance for GAE")

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    print(f"Reward Mode: {args.reward_mode}")
    print(f"Training Steps: {args.timesteps:,}")
    print(f"Max Episode Steps: {args.max_steps}")
    print(f"Seed: {args.seed}")

    env = make_env(
        reward_mode=args.reward_mode,
        seed=args.seed,
        max_steps=args.max_steps
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,

        # PPO hyperparameters
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
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    print("Starting training:")

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True
    )

    save_name = f"ppo_snake_{args.reward_mode}"
    save_path = os.path.join(args.modeldir, save_name)
    model.save(save_path)

    print(f"Model saved to: {save_path}")

    print("\nShort Test:")
    test_scores = []

    for i in range(5):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        test_scores.append(info['score'])
        print(f" Episode {i + 1}: Score={info['score']}, Steps={steps}, Reward={total_reward:.1f}")

    avg_score = sum(test_scores) / len(test_scores)
    print(f"\nAverage Score: {avg_score:.1f}")
    print(f"Best Score: {max(test_scores)}")

    env.close()


if __name__ == "__main__":
    main()