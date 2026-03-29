"""
agent.py — Tabular Q-learning agent

The Q-table maps every (state, action) pair to an expected future reward.
After each move the agent updates its estimate:

    Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') − Q(s,a)]

  α (alpha)  = learning rate   — how fast to update
  γ (gamma)  = discount factor — how much to value future rewards

Exploration vs exploitation (ε-greedy):
  High ε → explore randomly (early training)
  Low  ε → exploit the learned Q-table (later training)
  ε decays after every episode.
"""

import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        action_space,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
    ):
        self.action_space  = action_space
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = defaultdict(lambda: np.zeros(len(action_space)))
        self.episode_count = 0

    def choose_action(self, state):
        """ε-greedy: explore randomly or exploit best known action."""
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Apply the Q-learning update rule."""
        current_q   = self.q_table[state][action]
        best_next_q = 0.0 if done else np.max(self.q_table[next_state])
        target      = reward + self.gamma * best_next_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def end_episode(self):
        """Decay epsilon after each episode."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state):
        return int(np.argmax(self.q_table[state]))

    def q_values(self, state):
        return {a: round(float(self.q_table[state][a]), 2)
                for a in self.action_space}

    def states_visited(self):
        return len(self.q_table)

    def get_policy_grid(self, nrows, ncols):
        """2D grid of best action per cell (None if never visited)."""
        policy = [[None] * ncols for _ in range(nrows)]
        for r in range(nrows):
            for c in range(ncols):
                s = (r, c)
                if s in self.q_table:
                    policy[r][c] = self.best_action(s)
        return policy
