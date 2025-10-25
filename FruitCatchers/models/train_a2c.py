# train_a2c.py
from stable_baselines3 import A2C
from fruit_env_full import FruitCatchFullEnv
from train_agent import RewardLogger

env = FruitCatchFullEnv(persona="survivor")
a2c_callback = RewardLogger(log_dir="logs_csv", algo_name="A2C")

print("🚀 Training A2C model...")
model_a2c = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logs/A2C")
model_a2c.learn(total_timesteps=500000, callback=a2c_callback)
model_a2c.save("models/a2c_fruit")
print("A2C model saved → models/a2c_fruit.zip")
