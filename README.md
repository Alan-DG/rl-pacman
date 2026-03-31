# RL Pac-Man — Q-learning demo

A Pac-Man-style gridworld where a reinforcement learning agent learns to
navigate a maze from start to goal using **tabular Q-learning**.

Built as a demo for an Advanced Machine Learning presentation.

---

## What the agent does

The agent starts top-left `(1,1)` and must reach the goal cell `(8,8)`.
Pellets scattered through the maze give small bonus rewards when walked over,
but the **win condition is simply reaching the goal**.

The state is just `(row, col)` — a tiny state space — so Q-learning converges
fast and the learned policy is easy to inspect and explain.

There is no ghost. Scope was kept deliberately minimal so the focus stays on
RL fundamentals rather than environment complexity.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full 5-act presentation demo
python demo.py

# Jump straight to a specific act
python demo.py --act 3

# Slow everything down (useful on slower machines)
python demo.py --fps 4

# Headless training only (saves results/)
python src/train.py
```

---

## Demo structure

The demo runs as five acts. Press **SPACE** or **ENTER** to advance between
acts; **ESC** quits at any point.

| Act | Title | What it shows |
|-----|-------|---------------|
| 1 | Untrained agent | Pure random exploration — stumbling, no goal |
| 2 | Live training | Snapshots at episodes 1, 8, 20, 50, 100, 200 — watch it learn |
| 3 | Trained agent | Clean optimal run + Q-heatmap + policy arrows |
| 4 | New maze transfer | Same Q-table fails on MAZE_2; retrain → works again |
| 5 | Exploration vs exploitation | Same agent at ε = 1.0 → 0.5 → 0.15 → 0.0 |

---

## Controls (pygame window)

| Key | Action |
|-----|--------|
| `H` | Toggle Q-value heatmap (cell colour = learned value: blue → red) |
| `Q` | Toggle policy overlay (best learned action per cell) |
| `SPACE` | Pause / resume |
| `ESC` | Quit |

The heatmap and policy arrows work simultaneously. Both are most informative
after training converges (Act 3).

---

## Project structure

```
rl-pacman/
├── demo.py              ← run this for the presentation
├── requirements.txt
├── README.md
├── src/
│   ├── maze.py          ← two maze layouts, rewards, step logic
│   ├── agent.py         ← Q-learning: Q-table, ε-greedy, update rule
│   ├── train.py         ← training loop, snapshot rendering, KPI logging
│   └── renderer.py      ← pygame window, heatmap, policy overlay
└── results/             ← auto-generated on first training run
    ├── rewards.csv
    └── reward_curve.png
```

---

## Mazes

Two layouts are included, both 10×10 with the same start `(1,1)` and goal `(8,8)`.

**MAZE_1** — branching paths, diagonal route through the middle. Used for Acts 1–3 and Act 5.

**MAZE_2** — tight spiral layout. Optimal path winds right → down → left → down → right before reaching the goal, making it roughly twice as long as MAZE_1's route. Used in Act 4 to demonstrate that Q-learning memorises a specific layout rather than generalising.

---

## Reward structure

| Event | Reward |
|-------|--------|
| Each step taken | −1 |
| Walking over a pellet | +5 |
| Walking over a power pellet | +15 |
| Reaching the goal | +100 |
| Timeout (400 steps) | episode ends, no bonus |

The step penalty encourages short paths. The goal reward dominates, so the
agent always prioritises reaching the exit.

---

## How Q-learning works

The agent maintains a **Q-table** — a lookup table mapping every
`(state, action)` pair to an expected future reward.

After each action it updates its estimate:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]
```

| Symbol | Name | Value | Meaning |
|--------|------|-------|---------|
| `α` | Learning rate | 0.1 | How much to shift the estimate each update |
| `γ` | Discount factor | 0.95 | How much future rewards are worth vs immediate |
| `r` | Reward | varies | What the agent received for this action |
| `max Q(s', a')` | Best future Q | — | Best known value reachable from next state |

**Exploration vs exploitation (ε-greedy):** the agent starts fully random
(`ε = 1.0`) and gradually shifts to exploiting its learned Q-table
(`ε` decays to `0.05`). Act 5 demonstrates this directly: the same trained
agent runs at ε = 1.0, 0.5, 0.15, and 0.0 — path length drops from ~200
steps down to the optimal 14.

---

## KPIs

`results/rewards.csv` — one row per training episode:

| Column | Description |
|--------|-------------|
| `episode` | Episode number |
| `total_reward` | Sum of all rewards in that episode |
| `steps` | Steps taken before win or timeout |
| `epsilon` | Exploration rate at end of episode |
| `pellets` | Number of pellets collected |
| `won` | `1` if the agent reached the goal, `0` otherwise |

`results/reward_curve.png` — smoothed reward per episode (left) and rolling
win rate (right). Both trend upward as training progresses.

---

## Hyperparameters

Defined in `src/agent.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.1 | Higher = faster updates, less stable |
| `gamma` | 0.95 | Higher = agent values long-term rewards more |
| `epsilon_decay` | 0.997 | Lower = stops exploring sooner |
| `epsilon_min` | 0.05 | Floor — retains 5% random exploration throughout |

`src/train.py` also accepts `initial_epsilon` and `maze_layout` overrides,
used internally by the demo for different training scenarios.

---

## Dependencies

- `pygame` — animated maze window
- `numpy` — Q-table operations
- `matplotlib` — reward curve plot
