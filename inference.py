"""
inference.py
------------
Standardized inference interface for the Cart Abandonment RL Environment.
Satisfies hackathon requirements:
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
  - Produces structured START / STEP / END logs
  - Runs full episodes against the deployed API
  - Uses trained Q-table if available, falls back to rule-based policy

Usage:
    # Local server, rule-based fallback
    python inference.py

    # Local server, use trained Q-table
    python train_and_save.py   # generates qtable_medium.pkl etc. first
    python inference.py

    # Against HF Spaces (Windows)
    set API_BASE_URL=https://yourname-cart-abandonment-env.hf.space
    set DIFFICULTY=hard
    python inference.py
"""

import os
import sys
import json
import time
import pickle
import logging
import requests
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL",  "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME",    "q-learning-v1")
HF_TOKEN     = os.environ.get("HF_TOKEN",      "")
DIFFICULTY   = os.environ.get("DIFFICULTY",    "medium")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "3"))
SEED         = int(os.environ.get("SEED",        "42"))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("inference")


def log(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": round(time.time(), 3), **kwargs}))


# ── Q-table loading ───────────────────────────────────────────────────────────

def load_qtable(difficulty: str):
    """
    Try to load the saved Q-table for the given difficulty.
    Returns None if file not found — inference falls back to rule-based policy.
    Run train_and_save.py first to generate these files.
    """
    path = f"qtable_{difficulty}.pkl"
    if not os.path.exists(path):
        logger.warning(f"[inference] Q-table not found: {path} — using rule-based fallback.")
        return None
    with open(path, "rb") as f:
        qtable = pickle.load(f)
    logger.info(f"[inference] Loaded Q-table: {path} ({len(qtable)} states)")
    return qtable


# ── State discretization (must match agent.py exactly) ───────────────────────

def discretize(obs: dict) -> tuple:
    """
    Convert API observation dict → Q-table state tuple.
    Must be identical to discretize_state() in agent.py.

    Dimensions:
      time_bin   : 0-2 / 3-5 / 6-8 / 9+         → 4 bins
      value_bin  : <750 / 750-1250 / >1250        → 3 bins
      discount   : 0 / 5 / 10 / 20               → as-is
      prod_disc  : 0 / 5 / 10 / 20               → as-is
      notified   : 0 or 1                         → as-is
    """
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


# ── Action selection ──────────────────────────────────────────────────────────

def select_action(obs: dict, qtable) -> tuple:
    """
    Returns (action, source) where source is 'qtable' or 'fallback'.

    If Q-table loaded and state is known → greedy learned policy.
    Otherwise → rule-based fallback so inference never crashes.
    """
    if qtable is not None:
        state = discretize(obs)
        if state in qtable:
            return int(np.argmax(qtable[state])), "qtable"
        # state unseen during training → fall through

    # Rule-based fallback
    t        = obs["time_since_abandon"]
    notified = obs.get("notified", False)
    discount = obs.get("discount_given", 0)

    if t < 2:
        action = 0                     # wait
    elif t < 4 and not notified:
        action = 1                     # notify once
    elif t < 7 and discount == 0:
        action = 3                     # discount 10%
    elif t >= 7 and discount < 20:
        action = 4                     # discount 20%
    else:
        action = 0                     # nothing left to do

    return action, "fallback"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def make_headers():
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def api_reset():
    r = requests.get(f"{API_BASE_URL}/reset", headers=make_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def api_step(action: int):
    r = requests.post(f"{API_BASE_URL}/step", headers=make_headers(),
                      json={"action": action}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(episode_num: int, qtable) -> dict:
    obs = api_reset()

    log("START",
        episode=episode_num,
        model=MODEL_NAME,
        difficulty=DIFFICULTY,
        cart_value=obs.get("cart_value"),
        product_discount=obs.get("product_discount"),
        policy="qtable" if qtable else "fallback",
    )

    step_num     = 0
    total_reward = 0.0

    while not obs.get("done", False):
        action, source = select_action(obs, qtable)
        obs = api_step(action)
        step_num     += 1
        total_reward += obs.get("reward", 0.0)

        log("STEP",
            episode=episode_num,
            step=step_num,
            action=action,
            action_source=source,      # 'qtable' or 'fallback' per step
            reward=round(obs.get("reward", 0.0), 4),
            discount_given=obs.get("discount_given"),
            notified=obs.get("notified"),
            done=obs.get("done"),
        )

    purchased = total_reward > -0.5

    log("END",
        episode=episode_num,
        model=MODEL_NAME,
        difficulty=DIFFICULTY,
        steps=step_num,
        total_reward=round(total_reward, 4),
        purchased=purchased,
        profit_score=round(max(total_reward, 0.0), 4),
    )

    return {
        "episode":      episode_num,
        "steps":        step_num,
        "total_reward": total_reward,
        "purchased":    purchased,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    qtable = load_qtable(DIFFICULTY)

    log("START",
        event_type="run",
        model=MODEL_NAME,
        api_base_url=API_BASE_URL,
        difficulty=DIFFICULTY,
        num_episodes=NUM_EPISODES,
        seed=SEED,
        policy="qtable" if qtable else "fallback",
        qtable_states=len(qtable) if qtable else 0,
    )

    results, errors = [], []

    for ep in range(1, NUM_EPISODES + 1):
        try:
            results.append(run_episode(ep, qtable))
        except requests.exceptions.ConnectionError:
            msg = f"Cannot reach API at {API_BASE_URL}. Start server: uvicorn app:app --port 8000"
            log("END", event_type="error", episode=ep, error=msg)
            errors.append(msg)
            break
        except Exception as e:
            log("END", event_type="error", episode=ep, error=str(e))
            errors.append(str(e))

    if results:
        win_rate     = sum(1 for r in results if r["purchased"]) / len(results)
        avg_reward   = sum(r["total_reward"] for r in results) / len(results)
        profit_ratio = sum(max(r["total_reward"], 0) for r in results) / len(results)

        log("END",
            event_type="summary",
            model=MODEL_NAME,
            difficulty=DIFFICULTY,
            episodes_run=len(results),
            win_rate=round(win_rate, 3),
            avg_reward=round(avg_reward, 4),
            profit_ratio=round(profit_ratio, 4),
            errors=len(errors),
        )

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()