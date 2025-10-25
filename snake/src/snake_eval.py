import argparse
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from snake_env import SnakeEnv


def run_episode(model, reward_mode="survival", render=False, max_steps=5000, seed=None):

    env = SnakeEnv(
        render_mode="human" if render else None,
        reward_mode=reward_mode,
        max_steps=max_steps,
        seed=seed
    )
    obs, info = env.reset()
    done = trunc = False

    ep_reward = 0.0
    steps = 0
    actions = []
    food_eaten = 0
    steps_per_food = []
    steps_since_last_food = 0

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))

        prev_score = info.get('score', 0)
        obs, r, done, trunc, info = env.step(action)

        ep_reward += r
        steps += 1
        steps_since_last_food += 1

        if info.get('score', 0) > prev_score:
            food_eaten += 1
            steps_per_food.append(steps_since_last_food)
            steps_since_last_food = 0

    score = int(info.get("score", 0))
    length = int(info.get("length", 3))

    action_counts = [actions.count(i) for i in range(4)]

    avg_steps_per_food = np.mean(steps_per_food) if steps_per_food else 0

    env.close()

    return {
        "reward": float(ep_reward),
        "score": score,
        "length": length,
        "steps": steps,
        "food_eaten": food_eaten,
        "avg_steps_per_food": float(avg_steps_per_food),
        "crashed": int(done and not trunc),
        "truncated": int(trunc),
        "action_up": action_counts[0],
        "action_down": action_counts[1],
        "action_left": action_counts[2],
        "action_right": action_counts[3],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Snake agent")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (without .zip extension)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", type=int, default=0,
                        help="Whether to render episodes (1 for yes, 0 for no)")
    parser.add_argument("--reward_mode", type=str, default="survival",
                        choices=["survival", "length"],
                        help="Reward mode to use")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum steps per episode")
    args = parser.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(f"logs/snake_eval_{args.reward_mode}.csv"), exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    print(f"Episodes: {args.episodes}")
    print(f"Reward Mode: {args.reward_mode}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Rendering: {'Yes' if args.render else 'No'}")
    print("=" * 60)

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

        print(f"Score: {metrics['score']}, Length: {metrics['length']}, Steps: {metrics['steps']}")

    print("\nEvaluation:")

    rewards = [r["reward"] for r in rows]
    scores = [r["score"] for r in rows]
    lengths = [r["length"] for r in rows]
    steps = [r["steps"] for r in rows]
    food_eaten = [r["food_eaten"] for r in rows]
    avg_steps_per_food = [r["avg_steps_per_food"] for r in rows if r["avg_steps_per_food"] > 0]

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_score = float(np.mean(scores))
    max_score = max(scores)
    mean_length = float(np.mean(lengths))
    max_length = max(lengths)
    mean_steps = float(np.mean(steps))
    mean_food = float(np.mean(food_eaten))
    crash_rate = float(np.mean([r["crashed"] for r in rows]))
    timeout_rate = float(np.mean([r["truncated"] for r in rows]))
    mean_efficiency = float(np.mean(avg_steps_per_food)) if avg_steps_per_food else 0

    print(f"Episodes: {len(rows)}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Score: {mean_score:.2f} (Best: {max_score})")
    print(f"Mean Snake Length: {mean_length:.1f} (Best: {max_length})")
    print(f"Mean Episode Steps: {mean_steps:.1f}")
    print(f"Mean Food Eaten: {mean_food:.1f}")
    print(f"Mean Steps per Food: {mean_efficiency:.1f}")
    print(f"Crash Rate: {crash_rate * 100:.1f}%")
    print(f"Timeout Rate: {timeout_rate * 100:.1f}%")

    print("\nPerformance: ")
    print("-" * 40)
    expert = sum(1 for r in rows if r["score"] >= 20)
    good = sum(1 for r in rows if 10 <= r["score"] < 20)
    moderate = sum(1 for r in rows if 5 <= r["score"] < 10)
    beginner = sum(1 for r in rows if r["score"] < 5)

    print(f"Expert (≥20 food): {expert} episodes ({expert / len(rows) * 100:.1f}%)")
    print(f"Good (10-19 food): {good} episodes ({good / len(rows) * 100:.1f}%)")
    print(f"Moderate (5-9 food): {moderate} episodes ({moderate / len(rows) * 100:.1f}%)")
    print(f"Beginner (<5 food): {beginner} episodes ({beginner / len(rows) * 100:.1f}%)")

    total_up = sum(r["action_up"] for r in rows)
    total_down = sum(r["action_down"] for r in rows)
    total_left = sum(r["action_left"] for r in rows)
    total_right = sum(r["action_right"] for r in rows)
    total_actions = total_up + total_down + total_left + total_right

    print("\n Actions:")
    print(f"UP:    {total_up:6d} ({total_up / total_actions * 100:5.1f}%)")
    print(f"DOWN:  {total_down:6d} ({total_down / total_actions * 100:5.1f}%)")
    print(f"LEFT:  {total_left:6d} ({total_left / total_actions * 100:5.1f}%)")
    print(f"RIGHT: {total_right:6d} ({total_right / total_actions * 100:5.1f}%)")

    print("\n Survival Time")
    print("-" * 40)
    survival_times = sorted(steps)
    median_survival = survival_times[len(survival_times) // 2]
    q1_survival = survival_times[len(survival_times) // 4]
    q3_survival = survival_times[3 * len(survival_times) // 4]

    print(f"Median Survival: {median_survival} steps")
    print(f"25th Percentile: {q1_survival} steps")
    print(f"75th Percentile: {q3_survival} steps")
    print(f"Longest Survival: {max(steps)} steps")

    fieldnames = [
        "episode", "reward", "score", "length", "steps", "food_eaten",
        "avg_steps_per_food", "crashed", "truncated",
        "action_up", "action_down", "action_left", "action_right"
    ]

    with open(f"logs/snake_eval_{args.reward_mode}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()