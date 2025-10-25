# plot_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast

logs = {
    "PPO_10": "logs_csv/PPO_10_rewards.csv",
    "A2C": "logs_csv/A2C_rewards.csv",
    "PPO_lr5e5": "logs_csv/PPO_lr5e5_rewards.csv"
}

plt.figure(figsize=(10, 6))

for model_name, path in logs.items():
    if not os.path.exists(path):
        print(f"Missing: {path} — skipping.")
        continue

    df = pd.read_csv(path)

    if "reward" not in df.columns:
        print(f"Skipping {model_name} (no reward column).")
        continue

    # Clean and convert reward values 
    def parse_reward(x):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return float(val[0])
            return float(val)
        except:
            return None

    df["reward"] = df["reward"].apply(parse_reward)
    df = df.dropna(subset=["reward"])  # remove any bad rows

    # Smooth rewards
    df["smooth_reward"] = df["reward"].rolling(window=20, min_periods=1).mean()

    plt.plot(df["episode"], df["smooth_reward"], label=model_name, linewidth=2)

# Plot formatting 
plt.title("Model Performance Comparison – Average Reward per Episode", fontsize=14, weight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average Reward (smoothed)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save and show
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/model_comparison.png", dpi=300)
plt.show()

print("Plot saved → plots/model_comparison.png")
