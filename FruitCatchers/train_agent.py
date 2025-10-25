# train_agent.py
import os
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from fruit_env_full import FruitCatchFullEnv

# Custom callback to save rewards per episode 
class RewardLogger(BaseCallback):
    def __init__(self, log_dir, algo_name, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.algo_name = algo_name
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get('done'):
            self.episode_rewards.append(self.locals['rewards'])
            self.episode_lengths.append(self.locals['infos'][0].get('episode_length', 0))
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame({
            "episode": range(1, len(self.episode_rewards) + 1),
            "reward": self.episode_rewards,
            "length": self.episode_lengths
        })
        os.makedirs("logs_csv", exist_ok=True)
        df.to_csv(f"logs_csv/{self.algo_name}_rewards.csv", index=False)
        print(f"[Saved] Episode data for {self.algo_name} → logs_csv/{self.algo_name}_rewards.csv")


# Initialize environment 
env = FruitCatchFullEnv(persona="survivor")

# Train PPO 
ppo_callback = RewardLogger(log_dir="logs_csv", algo_name="PPO_10") # Running PPO_10 because it is the best game the AI played 
model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/PPO_10")
model_ppo.learn(total_timesteps=500000, callback=ppo_callback)
model_ppo.save("models/ppo_fruit")

# Train A2C 
a2c_callback = RewardLogger(log_dir="logs_csv", algo_name="A2C")
model_a2c = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model_a2c.learn(total_timesteps=500000, callback=a2c_callback)
model_a2c.save("models/a2c_fruit")

print("\n Training complete. Models and logs saved successfully.")
