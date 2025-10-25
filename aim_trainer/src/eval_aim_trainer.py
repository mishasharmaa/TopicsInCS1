import argparse
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from aim_trainer_env import AimTrainerEnv


def run_episode(model, reward_mode="survival", render=False, max_steps=5000, seed=None):

    env = AimTrainerEnv(
        render_mode="human" if render else None,
        reward_mode=reward_mode,
        max_steps=max_steps,
        seed=seed
    )
    obs, info = env.reset()
    done = trunc = False

    ep_reward = 0.0
    steps = 0
    actions_taken = []

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(action.copy())
        obs, r, done, trunc, info = env.step(action)
        ep_reward += r
        steps += 1

    # Episode-level metrics
    score = int(info.get("score", 0))
    accuracy = float(info.get("accuracy", 0.0))
    hits = int(info.get("hits", 0))
    misses = int(info.get("misses", 0))
    total_clicks = hits + misses

    # Calculate action statistics
    actions_array = np.array(actions_taken)
    action_variance = np.var(actions_array, axis=0) if len(actions_taken) > 1 else [0.0, 0.0]
    avg_action = np.mean(actions_array, axis=0) if len(actions_taken) > 0 else [0.5, 0.5]

    env.close()

    return {
        "reward": float(ep_reward),
        "score": score,
        "accuracy": accuracy,
        "steps": steps,
        "hits": hits,
        "misses": misses,
        "total_clicks": total_clicks,
        "crashed": int(done and not trunc),
        "truncated": int(trunc),
        "action_var_x": float(action_variance[0]),
        "action_var_y": float(action_variance[1]),
        "avg_action_x": float(avg_action[0]),
        "avg_action_y": float(avg_action[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Aim Trainer agent")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (without .zip extension)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--reward_mode", type=str, default="accuracy",
                        choices=["survival", "accuracy"],
                        help="Reward function to use")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", type=int, default=0,
                        help="Whether to render episodes (1 for yes, 0 for no)")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum steps per episode")
    args = parser.parse_args()


    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(f"logs/snake_eval_{args.reward_mode}.csv"), exist_ok=True)

    model = PPO.load(args.model_path)

    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Rendering: {'Yes' if args.render else 'No'}")


    rows = []
    for ep in range(1, args.episodes + 1):
        print(f"Running episode {ep}/{args.episodes}...", end=" ")

        metrics = run_episode(
            model,
            reward_mode=args.reward_mode,
            render=bool(args.render),
            max_steps=args.max_steps,
            seed=args.seed
        )
        metrics["episode"] = ep
        rows.append(metrics)

        print(f"Score: {metrics['score']}, Accuracy: {metrics['accuracy']:.1%}, Reward: {metrics['reward']:.2f}")

    print("\nEvaluation:")

    rewards = [r["reward"] for r in rows]
    scores = [r["score"] for r in rows]
    accuracies = [r["accuracy"] for r in rows]
    steps = [r["steps"] for r in rows]
    hits = [r["hits"] for r in rows]
    misses = [r["misses"] for r in rows]

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_accuracy = float(np.mean(accuracies))
    std_accuracy = float(np.std(accuracies))
    mean_steps = float(np.mean(steps))
    total_hits = sum(hits)
    total_misses = sum(misses)
    overall_accuracy = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
    crash_rate = float(np.mean([r["crashed"] for r in rows]))

    print(f"Episodes: {len(rows)}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Mean Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.1%}")
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({total_hits}/{total_hits + total_misses})")
    print(f"Mean Episode Length: {mean_steps:.1f} steps")
    print(f"Crash Rate: {crash_rate * 100:.1f}%")
    print(f"Total Targets Hit: {total_hits}")
    print(f"Total Targets Missed: {total_misses}")

    print("\nPerformance:")
    print("-" * 30)
    excellent = sum(1 for r in rows if r["score"] >= 140)
    good = sum(1 for r in rows if 0.6 <= r["score"] < 139)
    fair = sum(1 for r in rows if 0.4 <= r["score"] < 110)
    poor = sum(1 for r in rows if r["score"] < 75)

    print(f"Excellent (≥140): {excellent} episodes")
    print(f"Good (110-139): {good} episodes")
    print(f"Fair (75-110): {fair} episodes")
    print(f"Poor (50): {poor} episodes")

    action_vars_x = [r["action_var_x"] for r in rows]
    action_vars_y = [r["action_var_y"] for r in rows]

    print("\nActions")
    print("-" * 30)
    print(f"Avg Action Variance X: {np.mean(action_vars_x):.4f}")
    print(f"Avg Action Variance Y: {np.mean(action_vars_y):.4f}")

    fieldnames = [
        "episode", "reward", "score", "accuracy", "steps", "hits", "misses",
        "total_clicks", "crashed", "truncated", "action_var_x", "action_var_y",
        "avg_action_x", "avg_action_y"
    ]

    with open(f"logs/snake_eval_{args.reward_mode}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



if __name__ == "__main__":
    main()