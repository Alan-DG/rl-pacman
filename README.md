# RL Pac-Man — Q-learning demo

A Pac-Man-style gridworld where a reinforcement learning agent learns to navigate a maze, collect pellets, and avoid a ghost — using **tabular Q-learning**.

Built as a demo for an Advanced Machine Learning presentation.

---

## What this demonstrates

| Concept | Implementation |
|---|---|
| **Reinforcement Learning** | Agent learns by trial and error, no labelled data |
| **Q-learning** | Tabular Q-table updated after every action |
| **Exploration vs exploitation** | ε-greedy policy with epsilon decay |
| **Reward shaping** | +10 pellet, +30 power pellet, -100 ghost, -1/step |
| **Learning curve** | Reward increases as Q-table converges |

---

## Quick start

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Run the full demo (train + visual playback)
python demo.py

# 3. Train only (no window, saves results/)
python src/train.py

# 4. Train with more episodes for better performance
python demo.py --episodes 3000
```

---

## Controls (pygame window)

| Key | Action |
|---|---|
| `Q` | Toggle Q-policy overlay (shows best action per cell) |
| `SPACE` | Pause / resume |
| `ESC` | Quit |

---

## Project structure

```
rl-pacman/
├── demo.py              ← entry point for presentation
├── requirements.txt
├── README.md
├── src/
│   ├── maze.py          ← gridworld environment
│   ├── agent.py         ← Q-learning agent
│   ├── train.py         ← training loop + KPI logging
│   └── renderer.py      ← pygame visualisation
└── results/
    ├── rewards.csv       ← per-episode log (auto-generated)
    └── reward_curve.png  ← learning curve plot (auto-generated)
```

---

## How Q-learning works

The agent maintains a **Q-table** — a lookup table mapping every `(state, action)` pair to an expected future reward.

After each action it updates its estimate:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]
```

| Symbol | Name | Meaning |
|---|---|---|
| `α` | Learning rate | How fast to update (0.1) |
| `γ` | Discount factor | How much to value future rewards (0.95) |
| `r` | Reward | What the agent just received |
| `max Q(s', a')` | Best future value | Best known action from next state |

**Exploration vs exploitation:** the agent starts fully random (ε=1.0) and gradually becomes more greedy (ε decays to 0.05), exploiting what it has learned.

---

## KPIs

After training, `results/rewards.csv` contains per-episode:

| Column | Description |
|---|---|
| `episode` | Episode number |
| `total_reward` | Sum of rewards in that episode |
| `steps` | Steps taken |
| `epsilon` | Exploration rate at that point |
| `pellets` | Pellets collected |
| `won` | 1 if all pellets collected, else 0 |

The reward curve (`results/reward_curve.png`) visualises the learning progression — reward should increase and win rate should climb as training progresses.

---

## Hyperparameters

Tunable in `src/agent.py`:

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | 0.1 | Higher = faster learning, less stable |
| `gamma` | 0.95 | Higher = agent plans further ahead |
| `epsilon_decay` | 0.995 | Lower = explores less, exploits sooner |
| `epsilon_min` | 0.05 | Always keeps 5% random exploration |

---

## Dependencies

- `pygame` — animated maze window
- `numpy` — Q-table operations
- `matplotlib` — reward curve plot
