import argparse
import pygame
from stable_baselines3 import PPO
from snake_env import SnakeEnv


def main():
    parser = argparse.ArgumentParser(description="Watch trained Snake agent play")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model (without .zip extension)")
    parser.add_argument("--reward_mode", type=str, default="survival",
                        choices=["survival", "length"],
                        help="Reward function to use")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum steps per episode")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames per second for rendering")
    args = parser.parse_args()

    print(f"Model: {args.model_path}")
    print(f"Reward Mode: {args.reward_mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"FPS: {args.fps}")

    print("\nControls:")
    print("SPACE - Pause/Resume")
    print("R - Restart current episode")
    print("Q - Quit")
    print("ESC - Quit")

    # Load the model
    model = PPO.load(args.model_path)

    for episode in range(1, args.episodes + 1):
        print(f"\nEpisode {episode}/{args.episodes}")

        env = SnakeEnv(
            render_mode="human",
            reward_mode=args.reward_mode,
            max_steps=args.max_steps,
        )

        obs, info = env.reset()
        done = trunc = False
        paused = False
        step_count = 0
        total_reward = 0.0
        current_fps = args.fps
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
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        current_fps = min(120, current_fps + 5)
                        print(f"Speed increased: {current_fps} FPS")
                    elif event.key == pygame.K_MINUS:
                        current_fps = max(5, current_fps - 5)
                        print(f"Speed decreased: {current_fps} FPS")

            if not paused and not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                step_count += 1

                if step_count % 100 == 0:
                    print(
                        f"  Step {step_count}: Score={info['score']}, Length={info['length']}, Reward={total_reward:.1f}")

            env.render()

            pygame.time.Clock().tick(current_fps)

        final_score = info.get('score', 0)
        final_length = info.get('length', 3)

        print(f"Episode {episode} Complete!")
        print(f"Final Score: {final_score}")
        print(f"Final Length: {final_length}")
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Outcome: {'Crashed' if done else 'Timeout'}")

        if episode < args.episodes:
            print("   Starting next episode in 3 seconds...")
            pygame.time.wait(3000)

        env.close()

    print(f"\nAll {args.episodes} episodes completed!")


if __name__ == "__main__":
    main()