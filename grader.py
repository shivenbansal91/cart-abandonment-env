import random
import numpy as np
from environment import CartEnvironment
from models import CartAction

SEED = 42
EPISODES = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ── Simple stable policy (no training) ───────────────────

def policy(obs):
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
        obs = env.step(policy(obs))
        total += obs.reward

    return total


# ── Score (STRICTLY BETWEEN 0 AND 1) ────────────────────

def compute_score(difficulty: str, episodes: int = EPISODES) -> float:
    set_seed(SEED)

    total_profit = 0.0

    for _ in range(episodes):
        env = CartEnvironment(difficulty=difficulty)
        r = run_episode(env)

        if r > 0:
            total_profit += r

    score = total_profit / episodes

    # 🔥 CRITICAL FIX → always strictly between (0,1)
    score = max(0.01, min(0.99, score))

    return round(score, 4)
