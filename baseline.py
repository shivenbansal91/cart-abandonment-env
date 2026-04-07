"""
baseline.py — reproducible rule-based baseline. No training needed.
Usage: python baseline.py
"""

import random
import numpy as np
from environment import CartEnvironment
from models import CartAction

SEED = 42
EPISODES = 200


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def baseline_policy(obs):
    t = obs.time_since_abandon
    if t < 2:
        return CartAction(action=0)
    elif t < 4:
        return CartAction(action=1)
    elif t < 7:
        return CartAction(action=3)
    else:
        return CartAction(action=4)


def run_episode(env):
    obs = env.reset()
    total = 0.0
    while not obs.done:
        obs = env.step(baseline_policy(obs))
        total += obs.reward
    return total


def evaluate(difficulty, episodes=EPISODES):
    set_seed(SEED)
    rewards = []
    for _ in range(episodes):
        env = CartEnvironment(difficulty=difficulty)
        rewards.append(run_episode(env))
    positive = [r for r in rewards if r > 0]
    profit_ratio = round(sum(positive) / episodes, 4)
    return {
        "difficulty": difficulty,
        "profit_ratio": profit_ratio,
        "avg_reward": round(float(np.mean(rewards)), 4),
        "win_rate": round(len(positive) / episodes, 3),
        "min": round(float(np.min(rewards)), 4),
        "max": round(float(np.max(rewards)), 4),
    }


if __name__ == "__main__":
    print("\nCart Abandonment Env — Baseline Inference")
    print(f"Policy: Rule-based (no learning) | Seed: {SEED} | Episodes: {EPISODES}")
    print("=" * 65)
    print(f"  {'Difficulty':<10} {'Score':>8} {'Avg Reward':>12} {'Win Rate':>10} {'Min':>8} {'Max':>8}")
    print("-" * 65)

    for d in ["easy", "medium", "hard"]:
        r = evaluate(d)
        print(
            f"  {r['difficulty']:<10} {r['profit_ratio']:>8.4f} "
            f"{r['avg_reward']:>12.4f} {r['win_rate']:>10.3f} "
            f"{r['min']:>8.4f} {r['max']:>8.4f}"
        )

    print("=" * 65)
    print("Score = sum(positive rewards) / episodes. Rewards normalized to [0,1].")