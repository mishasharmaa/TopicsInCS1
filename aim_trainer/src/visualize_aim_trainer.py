import argparse
import pygame
from stable_baselines3 import PPO
from aim_trainer_env import AimTrainerEnv


def main():
    parser = argparse.ArgumentParser(description="Watch trained Aim Trainer agent play")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (without .zip extension)")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum steps per episode")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames per second for rendering")
    parser.add_argument("--reward_mode", type=str, default="accuracy",
                        choices=["survival", "accuracy"],
                        help="Reward function to use")

    args = parser.parse_args()

    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")

    print("Controls:")
    print("SPACE - Pause/Resume")
    print("R - Restart current episode")
    print("Q - Quit")
    print("ESC - Quit")


    model = PPO.load(args.model_path)

    for episode in range(1, args.episodes + 1):
        print(f"\nEpisode {episode}/{args.episodes}")

        env = AimTrainerEnv(
            render_mode="human",
            max_steps=args.max_steps,
            reward_mode=args.reward_mode
        )

        obs, info = env.reset()
        done = trunc = False
        paused = False
        step_count = 0
        total_reward = 0.0
        pygame.init()
        while not (done or trunc):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    print("\nExiting")
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_r:
                        print("Restarting episode")
                        obs, info = env.reset()
                        done = trunc = False
                        step_count = 0
                        total_reward = 0.0
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        env.close()
                        print("\nExiting")
                        return

            if not paused and not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                step_count += 1

                if step_count % 50 == 0:
                    print(f"Step {step_count}: Score={info['score']}, Accuracy={info['accuracy']:.1%}")

            env.render()

            pygame.time.Clock().tick(args.fps)

        final_score = info.get('score', 0)
        final_accuracy = info.get('accuracy', 0.0)

        print(f"Episode {episode} Complete!")
        print(f"Final Score: {final_score}")
        print(f"Final Accuracy: {final_accuracy:.1%}")
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Outcome: {'Crashed' if done else 'Time Limit'}")

        if episode < args.episodes:
            pygame.time.wait(3000)

        env.close()

    print(f"\nAll {args.episodes} episodes completed!")


if __name__ == "__main__":
    main()