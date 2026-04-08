import os
import sys
import requests
from openai import OpenAI

# ── Setup LLM (MANDATORY) ────────────────────────────────

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],   # 🔥 injected by hackathon
    api_key=os.environ["API_KEY"]
)

MODEL = "gpt-4o-mini"   # safe default

# ── Your ENV API (HF Space) ──────────────────────────────

ENV_URL = "https://shiven91-cart-abandonment-env.hf.space"

# ── Logging (required format) ────────────────────────────

def log(event, **kwargs):
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    print(f"[{event}] " + " ".join(parts), flush=True)

# ── LLM Action Decision ──────────────────────────────────

def get_action_from_llm(obs):
    prompt = f"""
You are an RL agent for cart abandonment.

Observation:
{obs}

Actions:
0 = wait
1 = notify
2 = discount_5
3 = discount_10
4 = discount_20

Return ONLY the action number (0-4).
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        action = int(response.choices[0].message.content.strip())
        return max(0, min(4, action))
    except:
        return 0  # fallback

# ── API Calls ────────────────────────────────────────────

def reset():
    return requests.get(f"{ENV_URL}/reset").json()

def step(action):
    return requests.post(f"{ENV_URL}/step", json={"action": action}).json()

# ── Episode Runner ───────────────────────────────────────

def run_episode(ep):
    try:
        obs = reset()
    except:
        log("START", episode=ep, status="api_failed")
        log("END", episode=ep, total_reward=0)
        return

    log("START", episode=ep)

    total = 0
    step_count = 0

    while not obs.get("done", False):
        action = get_action_from_llm(obs)

        try:
            obs = step(action)
        except:
            break

        reward = obs.get("reward", 0)
        total += reward
        step_count += 1

        log("STEP", episode=ep, step=step_count, action=action, reward=round(reward, 4))

    log("END", episode=ep, total_reward=round(total, 4))

# ── Main ─────────────────────────────────────────────────

def main():
    for ep in range(1, 3):
        run_episode(ep)

    sys.exit(0)

if __name__ == "__main__":
    main()
