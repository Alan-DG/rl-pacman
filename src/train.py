"""
train.py — Training loop + KPI logging

New parameters vs original:
  renderer         — pass an existing PacManRenderer to show live snapshots
  snapshot_episodes— dict {episode_number: fps} — which episodes to render live
  alpha/gamma/epsilon_decay/epsilon_min — hyperparameter overrides

Usage
-----
    python src/train.py                  # headless, 1500 episodes
    python src/train.py --episodes 300
    python src/train.py --render         # legacy: render every 100 eps
"""

import os, sys, argparse, csv, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from maze  import MazeEnv, GOAL
from agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def train(
    n_episodes=1500,
    # Legacy params (still work from CLI)
    render=False,
    render_every=100,
    verbose=True,
    maze_layout=None,
    # New: pass an existing renderer and specific snapshot episodes
    renderer=None,
    snapshot_episodes=None,   # dict {ep: fps}  e.g. {1:2, 10:4, 50:8}
    # Hyperparameter overrides
    alpha=0.1,
    gamma=0.95,
    epsilon_decay=0.997,
    epsilon_min=0.05,
    initial_epsilon=1.0,
):
    """
    Train a Q-learning agent and return (agent, episode_log).

    snapshot_episodes overrides render_every when provided.
    renderer can be an existing PacManRenderer; if None and render=True,
    one is created internally.
    """
    env   = MazeEnv(maze_layout)
    agent = QLearningAgent(
        action_space  = env.action_space,
        alpha         = alpha,
        gamma         = gamma,
        epsilon_decay = epsilon_decay,
        epsilon_min   = epsilon_min,
        epsilon       = initial_epsilon,
    )

    # Renderer setup
    internal_renderer = False
    if renderer is None and render:
        from renderer import PacManRenderer
        renderer = PacManRenderer(env)
        internal_renderer = True

    if renderer is not None:
        renderer.env    = env
        renderer._agent = agent

    log        = []
    start_time = time.time()
    win_count  = 0   # rolling window for banner

    for ep in range(1, n_episodes + 1):
        state  = env.reset()
        total_r= 0
        done   = False

        # Decide if this episode is rendered live
        if snapshot_episodes is not None:
            show_ep = ep in snapshot_episodes
            ep_fps  = snapshot_episodes.get(ep, 8)
        else:
            show_ep = render and (ep % render_every == 0)
            ep_fps  = 8

        if renderer and show_ep:
            renderer.fps = ep_fps
            renderer.env = env
            win_r = win_count / min(ep, 50) * 100 if ep > 1 else 0
            renderer.set_banner(
                f"Episode {ep}/{n_episodes}  |  "
                f"Win rate: {win_r:.0f}%  |  "
                f"ε = {agent.epsilon:.3f}  |  "
                f"γ = {gamma}  α = {alpha}"
            )

        while not done:
            action              = agent.choose_action(state)
            next_state, r, done, info = env.step(action)
            agent.update(state, action, r, next_state, done)
            state    = next_state
            total_r += r

            if renderer and show_ep:
                if not renderer.render(ep, total_r, agent.epsilon):
                    save_results(log)
                    if internal_renderer:
                        renderer.close()
                    return agent, log

        won = int(tuple(env.agent_pos) == GOAL)
        win_count = sum(e['won'] for e in log[-49:]) + won  # recompute cleanly
        agent.end_episode()

        log.append({
            "episode":      ep,
            "total_reward": round(total_r, 2),
            "steps":        info["steps"],
            "epsilon":      round(agent.epsilon, 4),
            "won":          won,
        })

        if verbose and ep % 100 == 0:
            recent = log[-100:]
            avg_r  = np.mean([e["total_reward"] for e in recent])
            win_r  = np.mean([e["won"]          for e in recent]) * 100
            print(
                f"Ep {ep:5d}/{n_episodes} | "
                f"Avg reward: {avg_r:7.1f} | "
                f"Win rate: {win_r:5.1f}% | "
                f"ε={agent.epsilon:.3f} | "
                f"{time.time()-start_time:.0f}s"
            )

    if internal_renderer:
        renderer.close()

    save_results(log)
    return agent, log


def save_results(log):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not log:
        return
    csv_path = os.path.join(RESULTS_DIR, "rewards.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    print(f"Saved → {csv_path}")
    plot_reward_curve(log)


def plot_reward_curve(log, window=50):
    window = min(window, max(len(log) // 2, 1))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rewards  = [e["total_reward"] for e in log]
    episodes = [e["episode"]      for e in log]
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    sx       = episodes[window - 1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Q-learning training — RL Pac-Man", fontsize=13, fontweight="bold")

    axes[0].plot(episodes, rewards,  alpha=0.2, color="#7F77DD", lw=0.8, label="Raw")
    axes[0].plot(sx, smoothed, color="#534AB7", lw=2, label=f"Avg (n={window})")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total reward")
    axes[0].set_title("Reward per episode"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    win_rate = [
        np.mean([e["won"] for e in log[max(0,i - window):i]]) * 100
        for i in range(window, len(log) + 1)
    ] if len(log) > window else []
    axes[1].plot(sx, win_rate, color="#1D9E75", lw=2)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Win rate (%)")
    axes[1].set_title(f"Win rate (rolling {window} eps)")
    axes[1].set_ylim(0, 105); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "reward_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def demo_run(agent, maze_layout=None, n=3):
    env           = MazeEnv(maze_layout)
    agent.epsilon = 0.0
    print("\n--- Greedy demo run ---")
    for i in range(1, n + 1):
        state = env.reset(); done = False; total = 0
        while not done:
            state, r, done, info = env.step(agent.choose_action(state))
            total += r
        won = "WON ✓" if tuple(env.agent_pos) == GOAL else "lost ✗"
        print(f"  Run {i}: {won} | reward={total:.0f} | "
              f"steps={info['steps']} | pellets={info['pellets_eaten']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",     type=int,  default=1500)
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--render-every", type=int,  default=100)
    args = parser.parse_args()
    print(f"Training {args.episodes} episodes...")
    agent, log = train(args.episodes, render=args.render,
                       render_every=args.render_every)
    demo_run(agent)
    print("\nDone. Check results/")
