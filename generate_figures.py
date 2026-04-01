"""
generate_figures.py — Generate all presentation figures for RL Pac-Man

Run after demo.py has been run at least once (so results/rewards.csv exists).
If the CSV is missing, a fresh agent is trained silently first.

Usage:
    python generate_figures.py

Outputs → results/figures/
    01_steps_to_goal.png
    02_epsilon_winrate.png
    03_reward_curve_annotated.png
    04_qtable_heatmap.png
    05_path_trace.png
"""

import os, sys, csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, 'src'))
sys.path.insert(0, _here)
from maze  import MazeEnv, MAZE_1, GOAL, AGENT_START
from agent import QLearningAgent
from train import train

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
CSV_PATH    = os.path.join(RESULTS_DIR, 'rewards.csv')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette — dark theme matching the pygame demo
# ---------------------------------------------------------------------------

DARK_BG   = '#0a0a16'
DARK_MID  = '#12121e'
ACCENT    = '#534AB7'
ACCENT_LT = '#7F77DD'
C_GREEN   = '#1D9E75'
C_GOLD    = '#FFD200'
C_RED     = '#FF5050'
TEXT_CLR  = '#DDDDFF'
DIM       = '#8888AA'

WALL_FC   = '#1E1EC8'
WALL_EC   = '#3C3CFF'
FLOOR_FC  = '#0A0A14'
FLOOR_EC  = '#161628'

Q_CMAP = LinearSegmentedColormap.from_list(
    'rl_heat', ['#0A0A46', '#0082D2', '#DCc80A', '#DC280A']
)

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   DARK_MID,
    'axes.edgecolor':   '#333355',
    'axes.labelcolor':  TEXT_CLR,
    'axes.titlecolor':  TEXT_CLR,
    'xtick.color':      DIM,
    'ytick.color':      DIM,
    'text.color':       TEXT_CLR,
    'grid.color':       '#222244',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'legend.facecolor': DARK_MID,
    'legend.edgecolor': '#333355',
})

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def savefig(fig, filename):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  Saved → {path}')


def rolling_mean(values, window):
    """Centred rolling mean, padded at edges."""
    out = []
    for i in range(len(values)):
        lo  = max(0, i - window + 1)
        out.append(float(np.mean(values[lo:i + 1])))
    return np.array(out)


def draw_maze_bg(ax, layout):
    """Draw walls and floor cells onto a matplotlib Axes."""
    nrows, ncols = len(layout), len(layout[0])
    for r in range(nrows):
        for c in range(ncols):
            if layout[r][c] == 1:
                fc, ec = WALL_FC,  WALL_EC
            else:
                fc, ec = FLOOR_FC, FLOOR_EC
            ax.add_patch(mpatches.Rectangle(
                (c, nrows - 1 - r), 1, 1,
                facecolor=fc, edgecolor=ec, linewidth=0.5
            ))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect('equal')
    ax.axis('off')


def run_episode_record(agent, env, greedy=False):
    """Run one episode and return the list of (row, col) positions visited."""
    saved = agent.epsilon
    if greedy:
        agent.epsilon = 0.0
    state = env.reset()
    path  = [tuple(env.agent_pos)]
    done  = False
    while not done:
        action = agent.choose_action(state)
        state, _, done, _ = env.step(action)
        path.append(tuple(env.agent_pos))
    agent.epsilon = saved
    return path


# ---------------------------------------------------------------------------
# Load or generate training log
# ---------------------------------------------------------------------------

def load_log():
    if not os.path.exists(CSV_PATH):
        print('  rewards.csv not found — training a fresh agent (takes ~1 s)...')
        _, log = train(300, verbose=False, maze_layout=MAZE_1)
        return log
    log = []
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            entry = {}
            for k, v in row.items():
                if k in ('episode', 'steps', 'won'):
                    entry[k] = int(float(v))
                else:
                    try:
                        entry[k] = float(v)
                    except ValueError:
                        entry[k] = v
            log.append(entry)
    return log


# ---------------------------------------------------------------------------
# Figure 1 — Steps to goal per episode
# ---------------------------------------------------------------------------

def fig_steps_to_goal(log):
    episodes = [e['episode']    for e in log]
    steps    = [e['steps']      for e in log]
    smooth   = rolling_mean(steps, 30)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Steps to Goal per Episode', fontsize=14, fontweight='bold')

    ax.plot(episodes, steps,   alpha=0.15, color=ACCENT_LT, lw=0.8, label='Raw')
    ax.plot(episodes, smooth,  color=ACCENT, lw=2.2, label='Rolling avg (30 eps)')

    # Timeout line
    ax.axhline(400, color=C_RED, lw=1.2, linestyle=':', alpha=0.8)
    ax.annotate(
        'Timeout (400 steps)\n— agent never finds goal',
        xy=(episodes[5], 400), xytext=(episodes[5] + 8, 355),
        color=C_RED, fontsize=9,
        arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2)
    )

    # First reliable wins — where rolling mean first drops below 100
    drop_idx = next((i for i, s in enumerate(smooth) if s < 100), None)
    if drop_idx:
        ax.axvline(episodes[drop_idx], color=C_GOLD, lw=1.2, linestyle='--', alpha=0.8)
        ax.annotate(
            f'First reliable wins\n(ep ≈ {episodes[drop_idx]})',
            xy=(episodes[drop_idx], smooth[drop_idx]),
            xytext=(episodes[drop_idx] + 8, 180),
            color=C_GOLD, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_GOLD, lw=1.2)
        )

    # Optimal line
    ax.axhline(14, color=C_GREEN, lw=1.2, linestyle=':', alpha=0.8)
    ax.annotate(
        'Optimal path: 14 steps',
        xy=(episodes[len(episodes) * 3 // 4], 14),
        xytext=(episodes[len(episodes) // 2], 55),
        color=C_GREEN, fontsize=9,
        arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.2)
    )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps taken')
    ax.set_ylim(-15, 440)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True)
    savefig(fig, '01_steps_to_goal.png')


# ---------------------------------------------------------------------------
# Figure 2 — Epsilon decay overlaid with rolling win rate
# ---------------------------------------------------------------------------

def fig_epsilon_winrate(log):
    episodes = [e['episode'] for e in log]
    epsilon  = [e['epsilon'] for e in log]
    win_rate = rolling_mean([e['won'] for e in log], 30)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle('Exploration vs Exploitation over Training', fontsize=14, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.set_facecolor(DARK_MID)

    l1, = ax1.plot(episodes, epsilon,  color=C_RED,   lw=2.2, label='ε  (exploration rate)')
    l2, = ax2.plot(episodes, win_rate, color=C_GREEN, lw=2.2, label='Win rate (rolling 30 eps)')

    # Crossover: win rate first exceeds 0.5
    cross = next((i for i, w in enumerate(win_rate) if w >= 0.5), None)
    if cross:
        ax1.axvline(episodes[cross], color=C_GOLD, lw=1.2, linestyle='--', alpha=0.85)
        ax1.annotate(
            f'Win rate passes 50%\nε ≈ {epsilon[cross]:.2f}',
            xy=(episodes[cross], epsilon[cross]),
            xytext=(episodes[cross] + 10, 0.65),
            color=C_GOLD, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_GOLD, lw=1.2)
        )

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Epsilon  (ε)', color=C_RED)
    ax2.set_ylabel('Win rate (rolling 30 eps)', color=C_GREEN)
    ax1.set_ylim(-0.05, 1.15)
    ax2.set_ylim(-0.05, 1.15)
    ax1.tick_params(axis='y', colors=C_RED)
    ax2.tick_params(axis='y', colors=C_GREEN)

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=9, loc='center right')
    ax1.grid(True)
    savefig(fig, '02_epsilon_winrate.png')


# ---------------------------------------------------------------------------
# Figure 3 — Annotated reward curve
# ---------------------------------------------------------------------------

def fig_reward_curve(log):
    window   = max(10, min(50, len(log) // 5))
    episodes = [e['episode']      for e in log]
    rewards  = [e['total_reward'] for e in log]
    smooth   = rolling_mean(rewards, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Total Reward per Episode', fontsize=14, fontweight='bold')

    ax.plot(episodes, rewards, alpha=0.15, color=ACCENT_LT, lw=0.8, label='Raw reward')
    ax.plot(episodes, smooth,  color=ACCENT, lw=2.2,
            label=f'Rolling avg ({window} eps)')

    # Flat bottom — first quarter where agent mostly times out
    flat_idx = next((i for i, s in enumerate(smooth) if s > -350), len(episodes) // 3)
    ax.annotate(
        'Flat region:\nagent times out\nevery episode',
        xy=(episodes[flat_idx // 2], smooth[flat_idx // 2]),
        xytext=(episodes[flat_idx // 2] + 12, -310),
        color=C_RED, fontsize=9,
        arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2)
    )

    # Inflection — reward first climbs above -100
    inflect = next((i for i, s in enumerate(smooth) if s > -100), None)
    if inflect:
        ax.annotate(
            'Inflection point:\nagent starts finding\nthe goal reliably',
            xy=(episodes[inflect], smooth[inflect]),
            xytext=(episodes[inflect] + 10, -220),
            color=C_GOLD, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_GOLD, lw=1.2)
        )

    # Theoretical max
    ax.axhline(86, color=C_GREEN, lw=1.2, linestyle=':', alpha=0.8)
    ax.annotate(
        'Theoretical max: +86\n(+100 goal − 14 steps)',
        xy=(episodes[int(len(episodes) * 0.75)], 86),
        xytext=(episodes[int(len(episodes) * 0.52)], 55),
        color=C_GREEN, fontsize=9,
        arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.2)
    )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward')
    ax.legend(fontsize=9)
    ax.grid(True)
    savefig(fig, '03_reward_curve_annotated.png')


# ---------------------------------------------------------------------------
# Figure 4 — Q-table heatmap: before vs after training
# ---------------------------------------------------------------------------

def fig_qtable_heatmap():
    env = MazeEnv(MAZE_1)
    nrows, ncols = env.nrows, env.ncols

    print('  Training "before" agent (3 episodes)...')
    agent_before, _ = train(3,   verbose=False, maze_layout=MAZE_1)
    print('  Training "after" agent (300 episodes)...')
    agent_after,  _ = train(300, verbose=False, maze_layout=MAZE_1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.suptitle('Q-Table Max Value per Cell  —  Before vs After Training',
                 fontsize=13, fontweight='bold')

    V_MIN, V_MAX = -50, 100

    for ax, agent, title in [
        (axes[0], agent_before, 'After 3 episodes  (almost untrained)'),
        (axes[1], agent_after,  'After 300 episodes  (converged)'),
    ]:
        draw_maze_bg(ax, env.maze_layout)

        for r in range(nrows):
            for c in range(ncols):
                if env.maze_layout[r][c] == 1:
                    continue
                s = (r, c)
                if s not in agent.q_table:
                    continue
                val = float(np.max(agent.q_table[s]))
                t   = np.clip((val - V_MIN) / (V_MAX - V_MIN), 0, 1)
                col = Q_CMAP(t)
                ax.add_patch(mpatches.Rectangle(
                    (c, nrows - 1 - r), 1, 1,
                    facecolor=col, edgecolor='none', alpha=0.88
                ))
                ax.text(c + 0.5, nrows - 0.5 - r, f'{val:.0f}',
                        ha='center', va='center', fontsize=6.5,
                        color='white' if t < 0.65 else '#111111')

        # Start / goal labels
        sr, sc = AGENT_START
        gr, gc = GOAL
        ax.text(sc + 0.5, nrows - 0.5 - sr, 'S',
                ha='center', va='center', fontsize=12,
                fontweight='bold', color=C_GOLD)
        ax.text(gc + 0.5, nrows - 0.5 - gr, 'G',
                ha='center', va='center', fontsize=12,
                fontweight='bold', color='#00E664')

        ax.set_title(title, fontsize=10, pad=8)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(
        cmap=Q_CMAP, norm=plt.Normalize(vmin=V_MIN, vmax=V_MAX)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.72, pad=0.02)
    cbar.set_label('Max Q-value  (expected future reward)', color=TEXT_CLR, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=DIM, fontsize=8)

    savefig(fig, '04_qtable_heatmap.png')


# ---------------------------------------------------------------------------
# Figure 5 — Path trace: random vs trained
# ---------------------------------------------------------------------------

def fig_path_trace():
    env = MazeEnv(MAZE_1)
    nrows = env.nrows

    # Untrained: one random episode (no Q-learning updates, pure exploration)
    agent_raw = QLearningAgent(env.action_space, epsilon=1.0)
    path_raw  = run_episode_record(agent_raw, env, greedy=False)

    # Trained: greedy run after 300 training episodes
    print('  Training agent for path trace (300 episodes)...')
    agent_trained, _ = train(300, verbose=False, maze_layout=MAZE_1)
    path_opt = run_episode_record(agent_trained, env, greedy=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.suptitle('Agent Path — Random vs Trained', fontsize=13, fontweight='bold')

    for ax, path, title in [
        (axes[0], path_raw, f'Untrained (random)  —  {len(path_raw)-1} steps'),
        (axes[1], path_opt, f'Trained (greedy)  —  {len(path_opt)-1} steps'),
    ]:
        draw_maze_bg(ax, env.maze_layout)

        xs = [c + 0.5 for (r, c) in path]
        ys = [nrows - 0.5 - r for (r, c) in path]

        # Gradient line along path (plasma colourmap: dark purple → yellow)
        n = len(path)
        for i in range(n - 1):
            t   = i / max(n - 2, 1)
            col = plt.cm.plasma(0.15 + 0.7 * t)
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color=col, lw=2.8, alpha=0.85, solid_capstyle='round')

        # Start marker
        ax.plot(xs[0], ys[0], 'o', color=C_GOLD, markersize=11, zorder=5)
        ax.text(xs[0], ys[0] + 0.35, 'S',
                ha='center', va='center', fontsize=8,
                fontweight='bold', color=C_GOLD)

        # Goal marker
        ax.plot(xs[-1], ys[-1], '*', color='#00E664', markersize=15, zorder=5)
        ax.text(xs[-1], ys[-1] + 0.35, 'G',
                ha='center', va='center', fontsize=8,
                fontweight='bold', color='#00E664')

        ax.set_title(title, fontsize=10, pad=8)

    # Shared gradient legend
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.72, pad=0.02, ticks=[0, 0.5, 1])
    cbar.set_label('Path progress  (start → end)', color=TEXT_CLR, fontsize=9)
    cbar.ax.set_yticklabels(['Start', 'Mid', 'End'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=DIM, fontsize=8)

    savefig(fig, '05_path_trace.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading training log...')
    log = load_log()
    print(f'  {len(log)} episodes loaded.\n')

    print('Generating figures...')
    fig_steps_to_goal(log)
    fig_epsilon_winrate(log)
    fig_reward_curve(log)
    fig_qtable_heatmap()
    fig_path_trace()

    print(f'\nDone. All figures saved to results/figures/')
