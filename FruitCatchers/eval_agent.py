# eval_agent.py
import time
import pygame
from stable_baselines3 import PPO
from fruit_env_full import FruitCatchFullEnv

def main():
    # Load trained PPO model 
    model = PPO.load("models/ppo_fruit")

    # Create the environment 
    env = FruitCatchFullEnv(render_mode=True)

    obs, _ = env.reset()
    score = 0

    print("AI is now playing the full Fruit Catchers game...")

    running = True
    while running:
        # Allow manual window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                env.close()
                break

        if not running:
            break

        # Predict next action from trained model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward

        # Game over handling
        if done or truncated:
            print(f"Round finished. Final Score: {score:.2f}")
            time.sleep(1.5)
            obs, _ = env.reset()
            score = 0

    env.close()

if __name__ == "__main__":
    main()