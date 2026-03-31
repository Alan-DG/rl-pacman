"""
demo.py — Full presentation demo  (run this during the presentation)

Five acts, SPACE to advance between each:

  Act 1 — Untrained agent         pure random, watch it stumble
  Act 2 — Live training           snapshots at key episodes, watch it learn
  Act 3 — Trained agent           clean solve + Q-heatmap + policy arrows
  Act 4 — New maze transfer       same Q-table fails on MAZE_2, then retrains
  Act 5 — Exploration vs exploit  same trained agent at ε=1.0→0.5→0.15→0.0

Usage
-----
    python demo.py             # full 5-act presentation
    python demo.py --act 2     # jump to a specific act (1-5)
    python demo.py --fps 4     # slow everything down
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze     import MazeEnv, MAZE_1, MAZE_2, GOAL
from agent    import QLearningAgent
from train    import train
from renderer import PacManRenderer


# ---------------------------------------------------------------------------
# Snapshot schedules  {episode: fps}
# Chosen to show the most visually interesting moments of learning
# ---------------------------------------------------------------------------
SNAPSHOTS_MAZE1 = {
    1:   2,    # pure chaos — slow so audience can follow
    8:   3,    # first hints of a path forming
    20:  4,
    50:  6,    # noticeably better
    100: 10,   # mostly solved
    200: 18,   # clean and fast
}

SNAPSHOTS_MAZE2 = {1: 2, 10: 3, 30: 5, 80: 9, 200: 18}
SNAPSHOTS_BAD   = {1: 2, 50: 5, 150: 8, 280: 10}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(act_num, title):
    print(f"\n{'='*60}")
    print(f"  Act {act_num} — {title}")
    print(f"{'='*60}")


def _run_random_episodes(renderer, env, n=3, fps=3, steps_limit=90):
    """Show a random (untrained) agent. Returns False if window closed."""
    agent = QLearningAgent(env.action_space, epsilon=1.0)
    renderer._agent = agent
    renderer.fps    = fps
    for ep in range(1, n + 1):
        state = env.reset()
        done  = False
        steps = 0
        while not done and steps < steps_limit:
            action = agent.choose_action(state)
            state, _, done, _ = env.step(action)
            steps += 1
            if not renderer.render(ep, 0, 1.0):
                return False
    return True


# ---------------------------------------------------------------------------
# Acts
# ---------------------------------------------------------------------------

def act1_random(renderer):
    """Untrained agent — pure random exploration."""
    _header(1, "Untrained agent — random exploration")
    print("  No training yet. Every move is random.")
    print("  Watch it wander, bump walls, never find the goal reliably.")

    env = MazeEnv(MAZE_1)
    renderer.env = env
    renderer.set_banner("Act 1 — Untrained agent  (ε = 1.0, pure random)")

    if not _run_random_episodes(renderer, env, n=3, fps=3, steps_limit=90):
        return False

    return renderer.wait_for_space(
        "Act 2 — Live training",
        "Watch the agent learn episode by episode   →   SPACE to start"
    )


def act2_training(renderer):
    """Train on MAZE_1 with live snapshots at key episodes."""
    _header(2, "Live training — Maze 1")
    print("  Training 300 episodes. Snapshots render at key milestones.")
    print("  Watch the reward climb and the path get more deliberate.\n")

    agent, log = train(
        n_episodes        = 300,
        renderer          = renderer,
        snapshot_episodes = SNAPSHOTS_MAZE1,
        verbose           = True,
        maze_layout       = MAZE_1,
    )

    wr = np.mean([e["won"] for e in log[-50:]]) * 100
    print(f"\n  Final win rate (last 50 eps): {wr:.0f}%")
    print(f"  Q-table size: {agent.states_visited()} states")

    if not renderer.wait_for_space(
        "Act 3 — Trained agent",
        "[H] heatmap   [Q] policy arrows   →   SPACE"
    ):
        return None
    return agent


def act3_trained(renderer, agent):
    """Show the trained agent playing cleanly with overlays."""
    _header(3, "Trained agent showcase")
    print("  Agent is now greedy (ε ≈ 0). Watch the clean optimal path.")
    print("  Press [H] for Q-value heatmap, [Q] for policy arrows.")

    env = MazeEnv(MAZE_1)
    renderer.env    = env
    renderer._agent = agent
    renderer.set_banner(
        "Act 3 — Trained agent  |  [H] heatmap  [Q] arrows  [SPACE] pause"
    )
    renderer.demo_play(agent, n_episodes=4, fps=5)

    return renderer.wait_for_space(
        "Act 4 — New maze, same agent",
        "Does the Q-table transfer to a new layout?   →   SPACE"
    )


def act4_transfer(renderer, agent):
    """Transfer failure on MAZE_2, then retrain."""
    _header(4, "Transfer test — Maze 2")
    print("  Same Q-table, completely different maze layout (spiral).")
    print("  The MAZE_1 agent tries to go DOWN at cell (1,4).")
    print("  In MAZE_2 that direction is a wall → agent gets stuck.")

    # Phase A: old agent on new maze
    env2 = MazeEnv(MAZE_2)
    renderer.env    = env2
    renderer._agent = agent
    saved_eps       = agent.epsilon
    agent.epsilon   = 0.05
    renderer.set_banner(
        "Act 4a — MAZE_1 agent on MAZE_2  |  Q-table has no knowledge of this layout"
    )
    print("\n  Running MAZE_1 agent on MAZE_2...")
    renderer.demo_play(agent, n_episodes=3, fps=4)
    agent.epsilon = saved_eps

    if not renderer.wait_for_space(
        "Retrain from scratch on Maze 2",
        "Same algorithm, new Q-table   →   SPACE"
    ):
        return False

    # Phase B: fresh agent trained on MAZE_2
    _header("4b", "Retraining on Maze 2")
    print("  Fresh agent learning the spiral path from scratch.")

    agent2, _ = train(
        n_episodes        = 300,
        renderer          = renderer,
        snapshot_episodes = SNAPSHOTS_MAZE2,
        verbose           = True,
        maze_layout       = MAZE_2,
    )

    renderer.set_banner("Act 4b — Retrained on MAZE_2  |  now it knows the spiral")
    renderer.demo_play(agent2, n_episodes=3, fps=5)

    return renderer.wait_for_space(
        "Act 5 — Exploration vs exploitation",
        "Same trained agent, different ε values   →   SPACE"
    )


def act5_explore_vs_exploit(renderer, agent):
    """Show the trained agent at different epsilon values."""
    _header(5, "Exploration vs Exploitation")
    print("  The same trained agent, deployed with different epsilon values.")
    print("  ε=1.0 → always random (as if untrained).")
    print("  ε=0.0 → fully greedy, takes the optimal path every time.")

    env = MazeEnv(MAZE_1)
    renderer.env = env

    scenarios = [
        (1.0,  "ε = 1.0  —  always random  (same as the untrained agent in Act 1)"),
        (0.5,  "ε = 0.5  —  50% random, 50% learned policy"),
        (0.15, "ε = 0.15  —  mostly learned, a little noise"),
        (0.0,  "ε = 0.0  —  fully greedy  (pure exploitation, optimal path)"),
    ]

    for i, (eps, label) in enumerate(scenarios):
        agent.epsilon = eps
        renderer.set_banner(f"Act 5 — {label}")
        renderer.demo_play(agent, n_episodes=3, fps=5)
        if i < len(scenarios) - 1:
            next_eps = scenarios[i + 1][0]
            if not renderer.wait_for_space(
                f"Next: ε = {next_eps}",
                "SPACE to continue"
            ):
                return

    renderer.wait_for_space(
        "Exploration drives learning — exploitation drives performance",
        "Demo finished — thank you!"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Pac-Man — presentation demo")
    parser.add_argument("--act", type=int, default=0,
                        help="Jump to a specific act (1-5). Default: run all.")
    parser.add_argument("--fps", type=int, default=0,
                        help="Cap all fps values (useful on slower machines).")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL Pac-Man — Q-learning  |  Presentation Demo")
    print("=" * 60)
    print("\n  Controls during any episode:")
    print("    [Q]     policy arrows (best action per cell)")
    print("    [H]     Q-value heatmap (blue = low value, red = high)")
    print("    [SPACE] pause / resume")
    print("    [ESC]   quit")
    print("\n  SPACE or ENTER advances between acts.")

    if args.fps > 0:
        for d in (SNAPSHOTS_MAZE1, SNAPSHOTS_MAZE2, SNAPSHOTS_BAD):
            for k in d:
                d[k] = min(d[k], args.fps)

    env      = MazeEnv(MAZE_1)
    renderer = PacManRenderer(env, fps=6)
    start    = args.act if args.act in range(1, 6) else 1

    try:
        agent = None

        if start <= 1:
            if not act1_random(renderer):
                renderer.close(); return

        if start <= 2:
            agent = act2_training(renderer)
            if agent is None:
                renderer.close(); return
        else:
            print("  (Training silently to get a baseline agent...)")
            agent, _ = train(300, verbose=False, maze_layout=MAZE_1)
            renderer._agent = agent

        if start <= 3:
            if not act3_trained(renderer, agent):
                renderer.close(); return

        if start <= 4:
            if not act4_transfer(renderer, agent):
                renderer.close(); return

        if start <= 5:
            act5_explore_vs_exploit(renderer, agent)

    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    finally:
        renderer.close()

    print("\nDemo finished. Check results/ for reward_curve.png and rewards.csv")


if __name__ == "__main__":
    main()
