# eval_agent.py
import sys
import time
import pygame
from stable_baselines3 import PPO, A2C
from fruit_env_full import FruitCatchFullEnv

def main():
    # Command-line argument handling
    if len(sys.argv) < 2:
        print("Usage: python eval_agent.py [ppo_10 | a2c | ppo_lr5e5]")
        return

    model_name = sys.argv[1].lower()

    # Map argument to model file paths
    model_paths = {
        "ppo_10": "models/ppo_fruit_10.zip",   # my best PPO model
        "a2c": "models/a2c_fruit.zip",
        "ppo_lr5e5": "models/ppo_fruit_lr5e5.zip"
    }

    if model_name not in model_paths:
        print(f"Unknown model '{model_name}'. Choose one of: {list(model_paths.keys())}")
        return

    model_path = model_paths[model_name]

    # Load the correct model (PPO or A2C)
    if "a2c" in model_name:
        model = A2C.load(model_path)
    else:
        model = PPO.load(model_path)

    # Create the environment 
    env = FruitCatchFullEnv(render_mode=True)
    obs, _ = env.reset()
    score = 0

    print(f"\n🎮 Running {model_name.upper()} model...\n")
    print("Close the window or press [X] to stop.\n")

    running = True
    while running:
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

        if done or truncated:
            print(f"Round finished. Final Score: {score:.2f}")
            time.sleep(1.5)
            obs, _ = env.reset()
            score = 0

    env.close()

if __name__ == "__main__":
    main()
