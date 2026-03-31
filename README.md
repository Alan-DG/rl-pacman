# RL Pac-Man — Q-learning demo

A Pac-Man-style gridworld where a reinforcement learning agent learns to
navigate a maze from start to goal using **tabular Q-learning**.

Built as a demo for an Advanced Machine Learning presentation.

---

## What the agent does

The agent starts top-left `(1,1)` and must reach the goal cell bottom-right
`(8,8)`. Pellets are scattered through the maze and give small bonus rewards
when walked over, but the **win condition is simply reaching the goal**.

This keeps the state space small — `state = (row, col)` — so Q-learning
converges in under a second and the learned policy is easy to inspect and
explain.

There is no Pac-Man ghost. The scope was deliberately kept minimal so the focus
stays on the RL fundamentals rather than environment complexity.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Train + watch the demo (recommended)
python demo.py

# Train only, headless (saves results/)
python src/train.py

# More episodes for a sharper policy
python demo.py --episodes 3000

# Slower playback for presentations
python demo.py --fps 3
```

---

## Controls (pygame window)

| Key | Action |
|---|---|
| `Q` | Toggle policy overlay — shows the best learned action per cell |
| `SPACE` | Pause / resume |
| `ESC` | Quit |

The policy overlay is the most useful thing to show during a presentation:
every floor cell gets a directional arrow showing what the agent has learned
to do from that position.

---

## Project structure

```
rl-pacman/
├── demo.py              ← run this for the presentation
├── requirements.txt
├── README.md
├── src/
│   ├── maze.py          ← gridworld: layout, rewards, step logic
│   ├── agent.py         ← Q-learning: Q-table, ε-greedy, update rule
│   ├── train.py         ← training loop, KPI logging, reward plot
│   └── renderer.py      ← pygame visualisation
└── results/             ← auto-generated when you run training
    ├── rewards.csv
    └── reward_curve.png
```

---

## Reward structure

| Event                       | Reward                 |
|-----------------------------|------------------------|
| Each step taken             | −1                     |
| Walking over a pellet       | +5                     |
| Walking over a power pellet | +15                    |
| Reaching the goal           | +100                   |
| Timeout (300 steps)         | episode ends, no bonus |

The step penalty encourages the agent to find a short path. The goal reward
dominates, so the agent always prioritises reaching the exit over collecting
every pellet.

---

## How Q-learning works

The agent maintains a **Q-table** — a lookup table mapping every
`(state, action)` pair to an expected future reward.

After each action it updates its estimate:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]
```

| Symbol          | Name            | Value  | Meaning                                        |
|-----------------|-----------------|--------|------------------------------------------------|
| `α`             | Learning rate   | 0.1    | How much to shift the estimate each update     |
| `γ`             | Discount factor | 0.95   | How much future rewards are worth vs immediate |
| `r`             | Reward          | varies | What the agent received for this action        |
| `max Q(s', a')` | Best future Q   | —      | Best known value reachable from next state     |

**Exploration vs exploitation (ε-greedy):** the agent starts fully random
(`ε = 1.0`) and gradually shifts toward exploiting its learned Q-table
(`ε` decays to `0.05`). This ensures it explores the maze before committing
to a fixed policy.

---

## KPIs

`results/rewards.csv` — one row per training episode:

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `episode`      | Episode number                                   |
| `total_reward` | Sum of all rewards in that episode               |
| `steps`        | Steps taken before win or timeout                |
| `epsilon`      | Exploration rate at end of episode               |
| `pellets`      | Number of pellets collected                      |
| `won`          | `1` if the agent reached the goal, `0` otherwise |

`results/reward_curve.png` — two plots: smoothed reward per episode and
rolling win rate. Both should trend upward as training progresses.

---

## Hyperparameters

Defined in `src/agent.py`:

| Parameter        | Default | Effect                                        |
|------------------|---------|-----------------------------------------------|
| `alpha`          | 0.1     | Higher = faster updates, less stable          |
| `gamma`          | 0.95    | Higher = agent values long-term rewards more  |
| `epsilon_decay`  | 0.997   | Lower = stops exploring sooner                |
| `epsilon_min`    | 0.05    | Floor — always keeps 5% random exploration    |

---

## Dependencies

- `pygame` — animated maze window
- `numpy` — Q-table operations
- `matplotlib` — reward curve plot
