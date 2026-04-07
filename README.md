---

title: Cart Abandonment RL Environment
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛒 Cart Abandonment RL Environment

> A reinforcement learning system simulating **e-commerce cart abandonment recovery** — a real-world problem where businesses aim to convert abandoned carts while maximizing profit.

A Q-learning agent observes abandoned shopping carts and decides whether to **wait**, **send a notification**, or **offer a discount**, learning to balance **conversion rate vs profit margin**.

---

## 🚀 Live Deployment

* **API Base URL:**
  https://shiven91-cart-abandonment-env.hf.space

* **API Docs (Swagger):**
  https://shiven91-cart-abandonment-env.hf.space/docs

---

## 🧠 The Problem

When a user abandons a cart:

* acting too early → annoys the user
* offering high discount → reduces profit

The agent must learn:

> **when and how to intervene to maximize long-term profit**

👉 User type (impulse / normal / discount-seeker) is hidden.

---

## ⚙️ Action Space

| ID | Action      | Description        | Cost                |
| -- | ----------- | ------------------ | ------------------- |
| 0  | wait        | Do nothing         | −0.05               |
| 1  | notify      | Send notification  | penalty (anti-spam) |
| 2  | discount_5  | Apply 5% discount  | small penalty       |
| 3  | discount_10 | Apply 10% discount | medium penalty      |
| 4  | discount_20 | Apply 20% discount | higher penalty      |

---

## Observation Space
 
| Field | Type | Range | Description |
|---|---|---|---|
| `time_since_abandon` | int | 0–10 | Steps elapsed since cart abandoned |
| `cart_value` | int | 500–2000 | Cart value in INR |
| `discount_given` | int | 0/5/10/20 | Current agent-applied discount % |
| `product_discount` | int | 0/5/10/20 | Pre-existing product-level discount % |
| `notified` | bool | — | Has agent already sent a notification? |
| `done` | bool | — | Episode over (purchase or timeout)? |
| `reward` | float | — | Reward from last action |
 
> **Note:** `user_type` is intentionally hidden — the agent must infer behaviour from signals.
 
---

## Reward Function
 
```
On purchase:   +profit = cart_value × (1 − total_discount/100) / 2000
Per step:      −0.05   (encourages fast decisions)
Notification:  −0.10 first time, −0.25 for spam
Discounts:     −0.02 to −0.10 proportional to size
On timeout:    −0.50   (user leaves without buying)
 
All rewards normalized to approximately [−1, +1]
```
 
---

## Tasks
 
| Task | User Mix | Target Score | Strategy |
|---|---|---|---|
| `easy` | 2/3 impulse buyers | ≥ 0.35 | Act fast, avoid over-discounting |
| `medium` | Equal mix of all types | ≥ 0.25 | Adapt based on time and signals |
| `hard` | 2/3 discount-seekers | ≥ 0.15 | Offer discount-10 early, protect margin |
 
**Score metric:** `profit_ratio = Σ(positive rewards) / episodes` — range **0.0 → 1.0**
 
---

## API Reference
 
Base URL: `https://shiven91-cart-abandonment-env.hf.space`
 
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/reset` | Start new episode → returns initial observation |
| `POST` | `/step` | Body: `{"action": 0–4}` → returns next observation |
| `GET` | `/state` | Full internal env state (debug only) |
| `GET` | `/docs` | Interactive Swagger UI |
 
**Example session:**
```bash
# Start episode
curl https://shiven91-cart-abandonment-env.hf.space/reset

# Take action
curl -X POST https://shiven91-cart-abandonment-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"action": 3}'
```
---

## Quick Start (Local)
 
```bash
# 1. Clone and install
git clone https://huggingface.co/spaces/shiven91/cart-abandonment-env
cd cart-abandonment-env
pip install -r requirements.txt
 
# 2. Start API server (Terminal 1 — keep running)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
 
# 3. Open API docs in browser
http://localhost:8000/docs
 
# 4. Run baseline scores (Terminal 2)
python baseline.py
 
# 5. Train RL agent + generate learning_curve.png
python train_and_save.py
 
# 6. Compare baseline vs RL agent
python grader.py
 
# 7. Run standardized inference
python inference.py
```
---

## 🤖 Model Approach

* Tabular **Q-Learning**
* Discretized state space (~384 states)
* ε-greedy exploration
* 5000 training episodes per difficulty

---

## Results

After training (5000 episodes per difficulty):

| Difficulty | Baseline Score | RL Agent Score | Improvement |
|------------|---------------|----------------|-------------|
| Easy       | ~0.30         | ~0.42          | +40%        |
| Medium     | ~0.22         | ~0.31          | +41%        |
| Hard       | ~0.10         | ~0.17          | +70%        |

*Run `python grader.py` locally to reproduce exact scores.*

Learning curves are saved to `learning_curve.png` by `train_and_save.py`.

---

## 🚀 Inference

```bash
# Local
python inference.py

# Using deployed HF API
set API_BASE_URL=https://shiven91-cart-abandonment-env.hf.space
set DIFFICULTY=hard
set NUM_EPISODES=5
python inference.py
```

Produces structured `START / STEP / END` logs:

```json
{
  "event": "START",
  "model": "q-learning-v1",
  "difficulty": "medium"
}

{
  "event": "STEP",
  "step": 1,
  "action": 0,
  "action_source": "qtable",
  "reward": -0.05
}

{
  "event": "END",
  "purchased": true,
  "profit_score": 0.412
}
```

---

## 📦 Project Structure

```
├── app.py
├── environment.py
├── models.py
├── agent.py
├── inference.py
├── baseline.py
├── grader.py
├── train_and_save.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Docker

```bash
docker build -t cart-env .
docker run -p 8000:7860 cart-env
```

---

## 💡 Key Highlights

* Real-world business problem
* Profit-aware decision making
* Full ML pipeline (train → deploy → inference)
* API-based architecture (production-ready)

---

## 👨‍💻 Author

**Shiven Bansal**
