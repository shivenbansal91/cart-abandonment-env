from environment import CartEnvironment
from models import CartAction
from agent import train, discretize_state
import numpy as np

print("=" * 52)
print("TEST 1: Manual episode walkthrough")
print("=" * 52)

env = CartEnvironment(difficulty="medium")
obs = env.reset()
print("RESET:", obs)

for step in range(15):
    if step == 2:
        action = CartAction(action=1)
    elif step == 4:
        action = CartAction(action=3)
    else:
        action = CartAction(action=0)

    obs = env.step(action)
    print(f"Step {step+1:>2}: reward={obs.reward:>8.4f} | done={obs.done} | discount={obs.discount_given} | notified={obs.notified}")
    if obs.done:
        print("  Episode finished.")
        break

print("\n" + "=" * 52)
print("TEST 2: Quick train + greedy episode")
print("=" * 52)

agent, rewards = train(difficulty="easy", episodes=1000, verbose=False)
print(f"States learned: {len(agent.q_table)}")
print(f"Avg reward last 200 ep: {np.mean(rewards[-200:]):.4f}")

env2 = CartEnvironment(difficulty="easy")
obs = env2.reset()
total = 0.0
print("\nGreedy episode (no exploration):")
while not obs.done:
    state = discretize_state(obs)
    action = int(np.argmax(agent.q_table[state])) if state in agent.q_table else 0
    obs = env2.step(CartAction(action=action))
    total += obs.reward
    print(f"  action={action} | reward={obs.reward:>8.4f} | done={obs.done}")

print(f"\nTotal episode reward: {total:.4f}")