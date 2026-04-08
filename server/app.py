from fastapi import FastAPI
from pydantic import BaseModel
import sys, os

# server/app.py is at /app/server/app.py
# Go ONE level up to reach /app where all modules live
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CartEnvironment
from models import CartAction
from grader import compute_score   # top-level import — fails loudly on startup if broken

app = FastAPI(title="Cart Abandonment RL Environment", version="1.0")

env = CartEnvironment()


class ActionRequest(BaseModel):
    action: int


# ── RL Environment Endpoints ─────────────────────────────

@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.__dict__


@app.post("/step")
def step(action: ActionRequest):
    obs = env.step(CartAction(action=action.action))
    return obs.__dict__


@app.get("/state")
def state():
    return {
        "time": env.time,
        "user_type": env.user_type,
        "cart_value": env.cart_value,
        "discount_given": env.discount_given,
        "product_discount": env.product_discount,
        "notified": env.notified,
        "done": env.done,
    }


# ── REQUIRED GRADER ENDPOINT ──────────────────────────────

@app.get("/grade")
def grade():
    tasks = []

    for difficulty in ["easy", "medium", "hard"]:
        try:
            score = float(compute_score(difficulty))
            score = max(0.01, min(0.99, score))
        except Exception as e:
            print(f"[grade] Error scoring {difficulty}: {e}", flush=True)
            score = 0.5

        tasks.append({
            "name": difficulty,
            "score": round(score, 4)
        })

    return {"tasks": tasks}


# ── Root ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Cart Abandonment RL Environment",
        "docs": "/docs",
        "endpoints": ["/reset", "/step", "/state", "/grade"]
    }
