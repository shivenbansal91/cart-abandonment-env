from fastapi import FastAPI
from pydantic import BaseModel
import sys, os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CartEnvironment
from models import CartAction

app = FastAPI(title="Cart Abandonment RL Environment", version="1.0")

env = CartEnvironment()


class ActionRequest(BaseModel):
    action: int


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


@app.get("/")
def root():
    return {
        "message": "Cart Abandonment RL Environment",
        "docs": "/docs",
        "endpoints": ["/reset", "/step", "/state"]
    }


# ✅ REQUIRED MAIN FUNCTION (for OpenEnv)
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()
