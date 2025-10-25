# train_agent_lr_variant.py
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from fruit_env_full import FruitCatchFullEnv

# Custom callback to log per-episode rewards 
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
        df.to_csv(f"logs_csv/PPO_lr5e5_rewards.csv", index=False)
        print("[Saved] PPO_lr5e5 reward log → logs_csv/PPO_lr5e5_rewards.csv")

# Train PPO with tweaked learning rate 
env = FruitCatchFullEnv(persona="survivor")

print("Training PPO variant with lower learning rate (5e-5)...")
ppo_lr_callback = RewardLogger(log_dir="logs_csv", algo_name="PPO_lr5e5")

model_ppo_lr = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=5e-5,     # hyperparameter tweak
    tensorboard_log="./logs/PPO_lr5e5"
)

model_ppo_lr.learn(total_timesteps=500000, callback=ppo_lr_callback)
model_ppo_lr.save("models/ppo_fruit_lr5e5")

print("PPO (learning-rate variant) training complete → models/ppo_fruit_lr5e5.zip")
