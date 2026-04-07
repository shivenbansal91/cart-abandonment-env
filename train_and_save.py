"""
train_and_save.py
-----------------
Trains Q-learning agent on all 3 difficulties, saves Q-tables,
and generates learning_curve.png — your main visual for the hackathon.

Usage: python train_and_save.py
Outputs: qtable_easy.pkl, qtable_medium.pkl, qtable_hard.pkl, learning_curve.png
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from agent import train

EPISODES = 5000
WINDOW = 200


def smooth(arr, w):
    return [np.mean(arr[max(0, i - w):i + 1]) for i in range(len(arr))]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Q-Learning Agent — Learning Curves (Cart Abandonment RL)",
        fontsize=13, fontweight="bold"
    )
    colors = {"easy": "#27ae60", "medium": "#e67e22", "hard": "#e74c3c"}

    for idx, difficulty in enumerate(["easy", "medium", "hard"]):
        print(f"\n{'='*52}")
        print(f"  Training: {difficulty.upper()} ({EPISODES} episodes)")
        print("=" * 52)

        agent, rewards = train(difficulty=difficulty, episodes=EPISODES, verbose=True)

        fname = f"qtable_{difficulty}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(agent.q_table, f)
        print(f"  Saved → {fname}  ({len(agent.q_table)} states)")

        ax = axes[idx]
        s = smooth(rewards, WINDOW)
        ax.plot(rewards, alpha=0.15, color=colors[difficulty], linewidth=0.4)
        ax.plot(s, color=colors[difficulty], linewidth=2.2, label=f"Smoothed (w={WINDOW})")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{difficulty.capitalize()} Difficulty", fontweight="bold", fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward (normalized)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        s_start = np.mean(rewards[:WINDOW])
        s_end = np.mean(rewards[-WINDOW:])
        ax.annotate(
            f"Early avg:  {s_start:+.3f}\nFinal avg:  {s_end:+.3f}\nGain: {s_end - s_start:+.3f}",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85)
        )

    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
    print("\n  Saved → learning_curve.png")
    plt.show()


if __name__ == "__main__":
    main()