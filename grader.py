"""
grader.py — compares rule-based baseline vs trained RL agent.
Scores are profit_ratio in [0.0, 1.0]. Higher = better.
"""

import random
import numpy as np
from environment import CartEnvironment
from models import CartAction
from agent import train, discretize_state

SEED = 42
EPISODES = 200   # more episodes → more stable scores
NORM = 2000.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ── Baseline policy ──────────────────────────────────────

def baseline_policy(obs):
    t = obs.time_since_abandon
    if t < 2:
        return CartAction(action=0)    # wait
    elif t < 4:
        return CartAction(action=1)    # notify once
    elif t < 7:
        return CartAction(action=3)    # discount 10%
    else:
        return CartAction(action=4)    # discount 20% last resort


def run_episode_baseline(env):
    obs = env.reset()
    total = 0.0
    while not obs.done:
        obs = env.step(baseline_policy(obs))
        total += obs.reward
    return total


# ── RL agent policy ──────────────────────────────────────

def run_episode_agent(env, agent):
    obs = env.reset()
    total = 0.0
    while not obs.done:
        state = discretize_state(obs)
        action = int(np.argmax(agent.q_table[state])) if state in agent.q_table else 0
        obs = env.step(CartAction(action=action))
        total += obs.reward
    return total


# ── Scoring ──────────────────────────────────────────────

def score_policy(run_fn, difficulty, episodes=EPISODES, **kwargs):
    """profit_ratio: sum of positive rewards / max possible."""
    set_seed(SEED)
    total_profit = 0.0
    for _ in range(episodes):
        env = CartEnvironment(difficulty=difficulty)
        r = run_fn(env, **kwargs)
        if r > 0:
            total_profit += r
    # rewards are already normalized by NORM in environment
    # so max per episode ≈ 1.0, total max ≈ episodes
    return round(total_profit / episodes, 4)


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nCart Abandonment RL — Grader Report")
    print(f"Seed: {SEED} | Episodes per task: {EPISODES}")
    print("=" * 65)
    print(f"  {'Difficulty':<10} {'Baseline':>10} {'RL Agent':>10} {'Delta':>8} {'Improvement':>12}")
    print("-" * 65)

    for difficulty in ["easy", "medium", "hard"]:
        b_score = score_policy(run_episode_baseline, difficulty)

        print(f"  Training RL [{difficulty}]...", end="", flush=True)
        agent, _ = train(difficulty=difficulty, episodes=5000, verbose=False)
        print(" done.")

        rl_score = score_policy(run_episode_agent, difficulty, agent=agent)

        delta = rl_score - b_score
        pct = (delta / max(b_score, 0.001)) * 100
        tag = "BETTER ✓" if delta > 0 else "worse ✗"

        print(f"  {difficulty:<10} {b_score:>10.4f} {rl_score:>10.4f} {delta:>+8.4f} {pct:>10.1f}%  {tag}")

    print("=" * 65)
    print("Metric : profit_ratio (normalized rewards, range 0→1)")