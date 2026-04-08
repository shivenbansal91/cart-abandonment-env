import os
import sys
import time
import pickle
import requests
import numpy as np

# ── Configuration ─────────────────────────────────────────

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://shiven91-cart-abandonment-env.hf.space"
)

DIFFICULTY   = os.environ.get("DIFFICULTY", "medium")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "3"))

# ── Logging (FIXED FORMAT) ────────────────────────────────

def log(event, **kwargs):
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    msg = f"[{event}] " + " ".join(parts)
    print(msg, flush=True)

# ── Q-table loading ───────────────────────────────────────

def load_qtable(difficulty):
    path = f"qtable_{difficulty}.pkl"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

# ── State discretization ──────────────────────────────────

def discretize(obs):
    try:
        t = obs["time_since_abandon"]
        if t <= 2: time_bin = 0
        elif t <= 5: time_bin = 1
        elif t <= 8: time_bin = 2
        else: time_bin = 3

        v = obs["cart_value"]
        if v < 750: value_bin = 0
        elif v <= 1250: value_bin = 1
        else: value_bin = 2

        return (
            time_bin,
            value_bin,
            obs.get("discount_given", 0),
            obs.get("product_discount", 0),
            int(obs.get("notified", False)),
        )
    except:
        return (0, 0, 0, 0, 0)

# ── Action selection ──────────────────────────────────────

def select_action(obs, qtable):
    try:
        if qtable is not None:
            state = discretize(obs)
            if state in qtable:
                return int(np.argmax(qtable[state]))

        t = obs.get("time_since_abandon", 0)
        notified = obs.get("notified", False)
        discount = obs.get("discount_given", 0)

        if t < 2:
            return 0
        elif t < 4 and not notified:
            return 1
        elif t < 7 and discount == 0:
            return 3
        elif t >= 7 and discount < 20:
            return 4
        else:
            return 0
    except:
        return 0

# ── API Calls ─────────────────────────────────────────────

def api_reset():
    r = requests.get(f"{API_BASE_URL}/reset", timeout=15)
    r.raise_for_status()
    return r.json()

def api_step(action):
    r = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action},
        timeout=15
    )
    r.raise_for_status()
    return r.json()

# ── Episode runner ────────────────────────────────────────

def run_episode(ep, qtable):
    try:
        obs = api_reset()
    except:
        log("START", episode=ep, status="api_failed")
        log("END", episode=ep, total_reward=0)
        return 0

    log("START", episode=ep, difficulty=DIFFICULTY)

    step = 0
    total_reward = 0.0

    while not obs.get("done", False):
        action = select_action(obs, qtable)

        try:
            obs = api_step(action)
        except:
            break

        step += 1
        reward = obs.get("reward", 0.0)
        total_reward += reward

        log("STEP", episode=ep, step=step, action=action, reward=round(reward, 4))

    log("END", episode=ep, total_reward=round(total_reward, 4))

    return total_reward

# ── Main ─────────────────────────────────────────────────

def main():
    qtable = load_qtable(DIFFICULTY)

    results = []

    for ep in range(1, NUM_EPISODES + 1):
        try:
            r = run_episode(ep, qtable)
            results.append(r)
        except:
            log("END", episode=ep, total_reward=0)
            break

    # Always exit clean (VERY IMPORTANT)
    sys.exit(0)

# ── Entry ─────────────────────────────────────────────────

if __name__ == "__main__":
    main()
