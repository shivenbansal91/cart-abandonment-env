import os
import sys
import json
import time
import pickle
import logging
import requests
import numpy as np

# ── Configuration ─────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME", "q-learning-v1")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
DIFFICULTY   = os.environ.get("DIFFICULTY", "medium")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "3"))
SEED         = int(os.environ.get("SEED", "42"))

# ── Logging ───────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("inference")

def log(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": round(time.time(), 3), **kwargs}))

# ── Q-table loading ───────────────────────────────────────

def load_qtable(difficulty: str):
    path = f"qtable_{difficulty}.pkl"
    if not os.path.exists(path):
        logger.warning(f"[inference] Q-table not found: {path} — using fallback.")
        return None
    try:
        with open(path, "rb") as f:
            qtable = pickle.load(f)
        logger.info(f"[inference] Loaded Q-table: {path}")
        return qtable
    except Exception as e:
        logger.warning(f"[inference] Failed to load Q-table: {e}")
        return None

# ── State discretization ──────────────────────────────────

def discretize(obs: dict) -> tuple:
    t = obs["time_since_abandon"]
    if t <= 2:    time_bin = 0
    elif t <= 5:  time_bin = 1
    elif t <= 8:  time_bin = 2
    else:         time_bin = 3

    v = obs["cart_value"]
    if v < 750:     value_bin = 0
    elif v <= 1250: value_bin = 1
    else:           value_bin = 2

    return (
        time_bin,
        value_bin,
        obs.get("discount_given", 0),
        obs.get("product_discount", 0),
        int(obs.get("notified", False)),
    )

# ── Action selection ──────────────────────────────────────

def select_action(obs: dict, qtable):
    if qtable is not None:
        state = discretize(obs)
        if state in qtable:
            return int(np.argmax(qtable[state])), "qtable"

    # fallback policy (never crash)
    t        = obs["time_since_abandon"]
    notified = obs.get("notified", False)
    discount = obs.get("discount_given", 0)

    if t < 2:
        action = 0
    elif t < 4 and not notified:
        action = 1
    elif t < 7 and discount == 0:
        action = 3
    elif t >= 7 and discount < 20:
        action = 4
    else:
        action = 0

    return action, "fallback"

# ── HTTP helpers (SAFE) ───────────────────────────────────

def make_headers():
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h

def api_reset():
    try:
        r = requests.get(f"{API_BASE_URL}/reset", headers=make_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Reset API failed: {e}")

def api_step(action: int):
    try:
        r = requests.post(
            f"{API_BASE_URL}/step",
            headers=make_headers(),
            json={"action": action},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Step API failed: {e}")

# ── Episode runner ────────────────────────────────────────

def run_episode(ep, qtable):
    obs = api_reset()

    log("START",
        episode=ep,
        model=MODEL_NAME,
        difficulty=DIFFICULTY,
        policy="qtable" if qtable else "fallback",
    )

    steps = 0
    total_reward = 0.0

    while not obs.get("done", False):
        action, source = select_action(obs, qtable)
        obs = api_step(action)

        steps += 1
        total_reward += obs.get("reward", 0.0)

        log("STEP",
            episode=ep,
            step=steps,
            action=action,
            action_source=source,
            reward=round(obs.get("reward", 0.0), 4),
            done=obs.get("done"),
        )

    log("END",
        episode=ep,
        total_reward=round(total_reward, 4),
        purchased=total_reward > -0.5,
    )

    return total_reward

# ── Main ─────────────────────────────────────────────────

def main():
    qtable = load_qtable(DIFFICULTY)

    log("START",
        event_type="run",
        model=MODEL_NAME,
        difficulty=DIFFICULTY,
        api=API_BASE_URL,
    )

    errors = 0

    for ep in range(1, NUM_EPISODES + 1):
        try:
            run_episode(ep, qtable)
        except Exception as e:
            log("END", event_type="error", episode=ep, error=str(e))
            errors += 1

    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
