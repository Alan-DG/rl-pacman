# RL Pac-Man — Q-learning presentation demo

A compact **reinforcement learning** project where a tabular **Q-learning** agent learns
to solve a Pac-Man-style maze by moving from a fixed start cell to a goal cell.

This project was built as a presentation/demo for an Advanced Machine Learning class.
It is intentionally small and visual, so the learning process is easy to explain live.

---

## Project overview

The environment is a 10×10 grid maze.

- **State**: the agent's grid position `(row, col)`
- **Actions**: `UP`, `DOWN`, `LEFT`, `RIGHT`
- **Objective**: reach the goal cell as efficiently as possible
- **Learning method**: tabular Q-learning with **ε-greedy** exploration

The simplicity is deliberate:
- the Q-table stays interpretable
- training converges quickly
- the effects of exploration, exploitation, and reward shaping are easy to show

There are **two maze layouts**:
- **MAZE_1**: the main training maze used for the core demo
- **MAZE_2**: a different layout used to show that tabular Q-learning does **not**
  generalise automatically to a new environment

---

## Current demo structure

The main presentation script is `demo.py`.

It runs as a **5-act interactive demo**:

| Act | Title | Purpose |
|-----|-------|---------|
| 1 | Untrained agent | Shows pure random behaviour before learning |
| 2 | Live training | Shows snapshot episodes during training so the audience can watch improvement |
| 3 | Trained agent | Shows a converged agent solving the maze cleanly |
| 4 | Transfer to new maze | Shows that the learned Q-table fails on a different layout, then retrains from scratch |
| 5 | Exploration vs exploitation | Shows the same trained agent under different ε values |

This structure is designed for a live class presentation where you want both:
1. a visual intuition for RL  
2. a concrete example of how Q-learning improves behaviour over time

---

## Environment details

Defined in `maze.py`.

### Start and goal
- **Start**: `(1, 1)`
- **Goal**: `(8, 8)`

### Action mapping
- `0 = UP`
- `1 = DOWN`
- `2 = LEFT`
- `3 = RIGHT`

### Episode ending conditions
An episode ends when:
- the agent reaches the goal, or
- the agent hits the timeout limit of **400 steps**

### Reward structure
The current reward system is:

| Event | Reward |
|------|--------|
| Any step | `-1` |
| Reaching the goal | `+100` |

That means a winning episode has:

`total_reward = 100 - number_of_steps`

So if the agent reaches the goal in the optimal **14 steps**, the theoretical maximum episode reward is:

`+86`

---

## How the agent learns

Defined in `agent.py`.

The project uses standard **tabular Q-learning**:

`Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]`

### Default hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `alpha` | `0.1` | Learning rate |
| `gamma` | `0.95` | Discount factor |
| `epsilon` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay` | `0.997` | Multiplicative ε decay after each episode |

### Exploration vs exploitation
The agent uses **ε-greedy** action selection:
- high ε → more random exploration
- low ε → more exploitation of learned Q-values

This is visualised directly in **Act 5** of the demo.

---

## Main files

The project currently consists of these main files:

```text
README.md
demo.py
generate_figures.py
requirements.txt

results/reward_curve.png
results/rewards.csv
results/figures/01_steps_to_goal.png
results/figures/02_epsilon_winrate.png
results/figures/03_reward_curve_annotated.png
results/figures/04_qtable_heatmap.png
results/figures/05_path_trace.png

slides/presentation_script.md
slides/rl-pacman.pptx

src/agent.py
src/maze.py
src/renderer.py
src/train.py
```

The rewards csv and plot are output after training and the results/figures are output from the csv through generate_figures.py.
See below for further details.

### File roles

- **`maze.py`**  
  Defines the environment, both maze layouts, reward logic, and step mechanics.

- **`agent.py`**  
  Implements the tabular Q-learning agent and ε-greedy policy.

- **`train.py`**  
  Runs training, logs KPIs per episode, and saves results to CSV / plot output.

- **`demo.py`**  
  Runs the full live presentation demo in 5 acts.

- **`renderer.py`**  
  Handles the pygame visualisation, overlays, controls, HUD, and transition screens.

- **`generate_figures.py`**  
  Generates polished static figures for slides/reporting based on training results.

---

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the full presentation demo
```bash
python demo.py
```

### Jump straight to a specific act
```bash
python demo.py --act 3
```

### Slow the demo down
```bash
python demo.py --fps 4
```

### Run training headlessly
```bash
python train.py --episodes 300
```

### Regenerate the presentation figures
```bash
python generate_figures.py
```

---

## Training outputs

Training saves outputs under `results/`.

### `results/rewards.csv`
Episode-level training log with the following fields:

| Column | Description |
|--------|-------------|
| `episode` | Episode number |
| `total_reward` | Sum of rewards collected in that episode |
| `steps` | Number of steps taken before goal or timeout |
| `epsilon` | Exploration rate after the episode |
| `won` | `1` if the goal was reached, otherwise `0` |

### `results/reward_curve.png`
Generated by `train.py`.  
Contains:
- reward per episode
- rolling win rate

### `results/figures/`
Generated by `generate_figures.py`.  
Current figure set:

- `01_steps_to_goal.png`
- `02_epsilon_winrate.png`
- `03_reward_curve_annotated.png`
- `04_qtable_heatmap.png`
- `05_path_trace.png`

These are intended for presentation slides and explainability.

---

## Static figures explained

`generate_figures.py` creates a slide-friendly visual summary of the training process.

### 1. Steps to goal per episode
Shows how the number of steps drops over training and approaches the optimal path length.

### 2. Exploration vs exploitation
Plots ε decay against rolling win rate to show how performance improves as random exploration decreases.

### 3. Total reward per episode
Shows the reward curve over time and marks the theoretical maximum based on the current reward design.

### 4. Q-table heatmap
Compares early vs trained Q-values across the maze to show how value concentrates along good routes.

### 5. Agent path trace
Compares a random/untrained path to a greedy/trained path.

---

## Controls during the pygame demo

Available in the main window:

| Key | Action |
|-----|--------|
| `Q` | Toggle policy arrows |
| `H` | Toggle Q-value heatmap |
| `SPACE` | Pause / resume |
| `S` | Skip current episode |
| `-` / `+` | Decrease / increase FPS |
| `ESC` | Quit |

There are also clickable on-screen buttons for the same controls.

---

## Important note about reproducibility

`generate_figures.py` reads from `results/rewards.csv` if it already exists.
So if you change the reward function or environment logic, you should regenerate the training results first.

Recommended workflow:

```bash
python train.py --episodes 300
python generate_figures.py
```

If needed, delete the old `results/rewards.csv` first so the figures cannot be generated from stale logs.

---

## Educational purpose of this project

This project is intentionally **not** a full Pac-Man clone and **not** a deep RL system.

Its purpose is to demonstrate, clearly and visually:

- how an RL agent improves through repeated interaction
- what a Q-table represents
- why exploration matters early in training
- how exploitation emerges later
- why tabular Q-learning learns a policy for a specific environment rather than true generalisation

That makes it a good teaching example for a short classroom presentation.

---

## Dependencies

Current code requires:

- `pygame`
- `numpy`
- `matplotlib`

`matplotlib` is needed for figure generation and saved training plots.
