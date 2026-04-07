import random
from models import CartAction, CartObservation

# Normalize all rewards to roughly [-1, +1] scale so Q-learning converges fast.
# Raw profit (500-2000) was 1000x bigger than penalties, making learning unstable.
NORM = 2000.0


class CartEnvironment:
    def __init__(self, difficulty="easy"):
        self.max_time = 10
        self.difficulty = difficulty
        self.time = 0
        self.user_type = None
        self.cart_value = 0
        self.discount_given = 0
        self.product_discount = 0
        self.notified = False   # NEW: track if notification was sent
        self.done = False

    def reset(self):
        self.time = 0
        self.discount_given = 0
        self.notified = False
        self.done = False

        if self.difficulty == "easy":
            self.user_type = random.choice(["impulse", "impulse", "normal"])
        elif self.difficulty == "medium":
            self.user_type = random.choice(["impulse", "normal", "discount"])
        else:  # hard
            self.user_type = random.choice(["discount", "discount", "normal"])

        self.cart_value = random.randint(500, 2000)
        self.product_discount = random.choice([0, 5, 10, 20])

        return self._get_obs(0.0)

    def step(self, action: CartAction):
        if self.done:
            return self._get_obs(0.0)

        self.time += 1
        reward = -0.05   # small step penalty (normalized)

        if action.action == 0:
            pass  # wait
        elif action.action == 1:
            if not self.notified:
                self.notified = True
                reward -= 0.05   # first notification is cheap
            else:
                reward -= 0.20   # spamming penalty
        elif action.action == 2:
            self.discount_given = 5
            reward -= 0.02
        elif action.action == 3:
            self.discount_given = 10
            reward -= 0.05
        elif action.action == 4:
            self.discount_given = 20
            reward -= 0.10

        bought = self._user_decision()

        if bought:
            self.done = True
            total_discount = self.product_discount + self.discount_given
            profit = self.cart_value * (1 - total_discount / 100)
            reward += profit / NORM   # normalized to ~0.25–1.0

        if self.time >= self.max_time:
            self.done = True
            reward -= 0.5   # timeout penalty (normalized)

        return self._get_obs(round(reward, 4))

    def _user_decision(self):
        if self.user_type == "impulse":
            return random.random() < 0.3

        elif self.user_type == "normal":
            total_discount = self.product_discount + self.discount_given
            return random.random() < 0.2 + (total_discount / 100)

        else:  # discount lover
            total_discount = self.product_discount + self.discount_given
            return total_discount >= 10 and random.random() < 0.5

    def _get_obs(self, reward: float):
        return CartObservation(
            time_since_abandon=self.time,
            cart_value=self.cart_value,
            discount_given=self.discount_given,
            product_discount=self.product_discount,
            notified=self.notified,   # NEW field
            done=self.done,
            reward=reward,
        )