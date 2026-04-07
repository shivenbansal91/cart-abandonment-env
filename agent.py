import numpy as np
import random
from environment import CartEnvironment
from models import CartAction


def discretize_state(obs):
    """
    State tuple for Q-table. More informative than before:
      time_bin    : 0-2 / 3-5 / 6-8 / 9+          → 4 bins
      value_bin   : low(<750) / mid / high(>1250)  → 3 bins
      discount    : 0 / 5 / 10 / 20                → 4 values
      prod_disc   : 0 / 5 / 10 / 20                → 4 values
      notified    : True / False                   → 2 values  ← NEW
    Total states: 4×3×4×4×2 = 384 (very manageable)
    """
    t = obs.time_since_abandon
    if t <= 2:
        time_bin = 0
    elif t <= 5:
        time_bin = 1
    elif t <= 8:
        time_bin = 2
    else:
        time_bin = 3

    v = obs.cart_value
    if v < 750:
        value_bin = 0
    elif v <= 1250:
        value_bin = 1
    else:
        value_bin = 2

    return (time_bin, value_bin, obs.discount_given, obs.product_discount, int(obs.notified))


class QLearningAgent:
    def __init__(
        self,
        n_actions=5,
        learning_rate=0.2,      # higher LR — faster convergence on small state space
        gamma=0.90,             # slightly lower — episode horizon is only 10 steps
        epsilon=1.0,
        epsilon_min=0.01,       # near-zero at end so eval is clean
        epsilon_decay=0.998,    # slower decay → more exploration before converging
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self._get_q(discretize_state(obs))))

    def learn(self, obs, action, reward, next_obs, done):
        state = discretize_state(obs)
        next_state = discretize_state(next_obs)
        current_q = self._get_q(state)[action]
        target = reward if done else reward + self.gamma * np.max(self._get_q(next_state))
        self._get_q(state)[action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(difficulty="easy", episodes=5000, verbose=True):
    """
    5000 episodes (was 2000). With decay=0.998, epsilon reaches ~0.01
    by episode 4600 — agent is nearly greedy by the end.
    """
    env = CartEnvironment(difficulty=difficulty)
    agent = QLearningAgent()
    rewards_log = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0

        while not obs.done:
            action = agent.choose_action(obs)
            next_obs = env.step(CartAction(action=action))
            agent.learn(obs, action, next_obs.reward, next_obs, next_obs.done)
            obs = next_obs
            total_reward += obs.reward

        agent.decay_epsilon()
        rewards_log.append(total_reward)

        if verbose and (ep + 1) % 500 == 0:
            avg = np.mean(rewards_log[-500:])
            print(f"  Ep {ep+1:>5} | Avg reward (last 500): {avg:>7.4f} | ε={agent.epsilon:.4f}")

    return agent, rewards_log


if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Training: {difficulty.upper()}")
        print('='*50)
        agent, _ = train(difficulty=difficulty, episodes=5000)
        print(f"Q-table states: {len(agent.q_table)}")