"""
demo.py — Full presentation demo  (run this during the presentation)

Five acts, SPACE to advance between each:

  Act 1 — Untrained agent         pure random, watch it stumble
  Act 2 — Live training           snapshots at key episodes, watch it learn
  Act 3 — Trained agent           clean solve + Q-heatmap + policy arrows
  Act 4 — New maze transfer       same Q-table fails on MAZE_2, then retrains
  Act 5 — Hyperparameter effect   γ=0.1 (shortsighted) vs γ=0.95 (good)

Usage
-----
    python demo.py             # full 5-act presentation
    python demo.py --act 1     # jump to a specific act (1-5)
    python demo.py --fps 4     # slow everything down
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze     import MazeEnv, MAZE_1, MAZE_2, GOAL
from agent    import QLearningAgent
from train    import train
from renderer import PacManRenderer


# Snapshot schedules — which episodes to show live and at what fps
# Chosen to highlight the most visually interesting moments of learning
SNAPSHOTS_MAZE1 = {
    1:   2,    # pure chaos — slow so audience can see it
    8:   3,    # starting to form a path occasionally
    20:  4,
    50:  6,    # getting good
    100: 10,   # mostly solved
    200: 18,   # clean and fast
}

SNAPSHOTS_MAZE2 = {
    1:   2,
    10:  3,
    30:  5,
    80:  9,
    200: 18,
}

SNAPSHOTS_BAD = {
    1:   2,
    50:  5,
    150: 8,
    280: 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(act_num, title):
    print(f"\n{'='*60}")
    print(f"  Act {act_num} — {title}")
    print(f"{'='*60}")


def _run_random_episodes(renderer, env, n=3, fps=3, steps_limit=80):
    """Show a random agent stumbling. Returns False if window closed."""
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
    print("  The agent has no knowledge. Every move is random.")
    print("  Notice: it wanders, bumps walls, never finds the goal reliably.")

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
    _header(2, "Live training on Maze 1")
    print("  Training 300 episodes. Snapshots render live at key milestones.")
    print("  Watch the reward climb and the path become more deliberate.\n")

    agent, log = train(
        n_episodes       = 300,
        renderer         = renderer,
        snapshot_episodes= SNAPSHOTS_MAZE1,
        verbose          = True,
        maze_layout      = MAZE_1,
    )

    last50 = log[-50:]
    wr = np.mean([e["won"] for e in last50]) * 100
    print(f"\n  Final win rate (last 50 eps): {wr:.0f}%")
    print(f"  Q-table size: {agent.states_visited()} states")

    if not renderer.wait_for_space(
        "Act 3 — Trained agent",
        "ε ≈ 0  |  [Q] policy arrows  [H] heatmap   →   SPACE"
    ):
        return None

    return agent


def act3_trained(renderer, agent):
    """Show the trained agent playing cleanly. Enable heatmap/arrows."""
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
        "Does the Q-table transfer?   →   SPACE"
    )


def act4_transfer(renderer, agent):
    """Transfer failure: same agent, different maze. Then retrain."""
    _header(4, "Transfer test — Maze 2")
    print("  Same Q-table, new maze layout.")
    print("  MAZE_1 agent learned: go DOWN at cell (1,4).")
    print("  In MAZE_2, (1,4) down is a WALL → agent gets stuck.")

    # --- Phase A: old agent on new maze ---
    env2 = MazeEnv(MAZE_2)
    renderer.env    = env2
    renderer._agent = agent
    old_eps         = agent.epsilon
    agent.epsilon   = 0.05   # tiny bit of exploration so it moves, but mostly stuck
    renderer.set_banner(
        "Act 4a — MAZE_1 agent on MAZE_2  |  Q-table has NO knowledge of this layout"
    )
    print("\n  Running MAZE_1 agent on MAZE_2...")
    renderer.demo_play(agent, n_episodes=3, fps=4)
    agent.epsilon = old_eps

    if not renderer.wait_for_space(
        "Now retrain on Maze 2",
        "New agent, new Q-table, same algorithm   →   SPACE"
    ):
        return False

    # --- Phase B: retrain on MAZE_2 ---
    _header("4b", "Retraining on Maze 2")
    print("  Training fresh agent on MAZE_2. Watch it learn the spiral path.")

    agent2, log2 = train(
        n_episodes       = 300,
        renderer         = renderer,
        snapshot_episodes= SNAPSHOTS_MAZE2,
        verbose          = True,
        maze_layout      = MAZE_2,
    )

    renderer.set_banner("Act 4b — Retrained on MAZE_2  |  now it knows the spiral path")
    renderer.demo_play(agent2, n_episodes=3, fps=5)

    return renderer.wait_for_space(
        "Act 5 — Hyperparameter effect",
        "γ = 0.1  (shortsighted)  vs  γ = 0.95  (standard)   →   SPACE"
    )


def act5_explore_vs_exploit(renderer, agent):
    """Show the trained agent at different epsilon values — exploration vs exploitation."""
    _header(5, "Exploration vs Exploitation")
    print("  The trained agent, deployed at different epsilon values.")
    print("  ε=1.0 → always random (as if untrained).")
    print("  ε=0.5 → 50% random, 50% learned policy.")
    print("  ε=0.0 → fully greedy, optimal learned path.")

    env = MazeEnv(MAZE_1)
    renderer.env = env

    scenarios = [
        (1.0,  "ε = 1.0  —  always random  (like the untrained agent in Act 1)"),
        (0.5,  "ε = 0.5  —  50% random, 50% learned  (mid-training behaviour)"),
        (0.15, "ε = 0.15  —  mostly learned, some noise  (late training)"),
        (0.0,  "ε = 0.0  —  fully greedy  (pure exploitation, optimal path)"),
    ]

    for eps, label in scenarios:
        agent.epsilon = eps
        renderer.set_banner(f"Act 5 — Exploration vs exploitation  |  {label}")
        renderer.demo_play(agent, n_episodes=3, fps=5)
        next_eps = scenarios[scenarios.index((eps, label)) + 1][0] if (eps, label) != scenarios[-1] else None
        if next_eps is not None:
            if not renderer.wait_for_space(
                f"Next: ε = {next_eps}",
                "SPACE to continue"
            ):
                return

    renderer.wait_for_space(
        "Act 5 complete — exploration drives learning, exploitation drives performance",
        "Demo finished — thank you!"
    )re random, watch it stumble
  Act 2 — Live training           snapshots at key episodes, watch it learn
  Act 3 — Trained agent           clean solve + Q-heatmap + policy arrows
  Act 4 — New maze transfer       same Q-table fails on MAZE_2, then retrains
  Act 5 — Hyperparameter effect   γ=0.1 (shortsighted) vs γ=0.95 (good)

Usage
-----
    python demo.py             # full 5-act presentation
    python demo.py --act 1     # jump to a specific act (1-5)
    python demo.py --fps 4     # slow everything down
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze     import MazeEnv, MAZE_1, MAZE_2, GOAL
from agent    import QLearningAgent
from train    import train
from renderer import PacManRenderer


# Snapshot schedules — which episodes to show live and at what fps
# Chosen to highlight the most visually interesting moments of learning
SNAPSHOTS_MAZE1 = {
    1:   2,    # pure chaos — slow so audience can see it
    8:   3,    # starting to form a path occasionally
    20:  4,
    50:  6,    # getting good
    100: 10,   # mostly solved
    200: 18,   # clean and fast
}

SNAPSHOTS_MAZE2 = {
    1:   2,
    10:  3,
    30:  5,
    80:  9,
    200: 18,
}

SNAPSHOTS_BAD = {
    1:   2,
    50:  5,
    150: 8,
    280: 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(act_num, title):
    print(f"\n{'='*60}")
    print(f"  Act {act_num} — {title}")
    print(f"{'='*60}")


def _run_random_episodes(renderer, env, n=3, fps=3, steps_limit=80):
    """Show a random agent stumbling. Returns False if window closed."""
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
    print("  The agent has no knowledge. Every move is random.")
    print("  Notice: it wanders, bumps walls, never finds the goal reliably.")

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
    _header(2, "Live training on Maze 1")
    print("  Training 300 episodes. Snapshots render live at key milestones.")
    print("  Watch the reward climb and the path become more deliberate.\n")

    agent, log = train(
        n_episodes       = 300,
        renderer         = renderer,
        snapshot_episodes= SNAPSHOTS_MAZE1,
        verbose          = True,
        maze_layout      = MAZE_1,
    )

    last50 = log[-50:]
    wr = np.mean([e["won"] for e in last50]) * 100
    print(f"\n  Final win rate (last 50 eps): {wr:.0f}%")
    print(f"  Q-table size: {agent.states_visited()} states")

    if not renderer.wait_for_space(
        "Act 3 — Trained agent",
        "ε ≈ 0  |  [Q] policy arrows  [H] heatmap   →   SPACE"
    ):
        return None

    return agent


def act3_trained(renderer, agent):
    """Show the trained agent playing cleanly. Enable heatmap/arrows."""
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
        "Does the Q-table transfer?   →   SPACE"
    )


def act4_transfer(renderer, agent):
    """Transfer failure: same agent, different maze. Then retrain."""
    _header(4, "Transfer test — Maze 2")
    print("  Same Q-table, new maze layout.")
    print("  MAZE_1 agent learned: go DOWN at cell (1,4).")
    print("  In MAZE_2, (1,4) down is a WALL → agent gets stuck.")

    # --- Phase A: old agent on new maze ---
    env2 = MazeEnv(MAZE_2)
    renderer.env    = env2
    renderer._agent = agent
    old_eps         = agent.epsilon
    agent.epsilon   = 0.05   # tiny bit of exploration so it moves, but mostly stuck
    renderer.set_banner(
        "Act 4a — MAZE_1 agent on MAZE_2  |  Q-table has NO knowledge of this layout"
    )
    print("\n  Running MAZE_1 agent on MAZE_2...")
    renderer.demo_play(agent, n_episodes=3, fps=4)
    agent.epsilon = old_eps

    if not renderer.wait_for_space(
        "Now retrain on Maze 2",
        "New agent, new Q-table, same algorithm   →   SPACE"
    ):
        return False

    # --- Phase B: retrain on MAZE_2 ---
    _header("4b", "Retraining on Maze 2")
    print("  Training fresh agent on MAZE_2. Watch it learn the spiral path.")

    agent2, log2 = train(
        n_episodes       = 300,
        renderer         = renderer,
        snapshot_episodes= SNAPSHOTS_MAZE2,
        verbose          = True,
        maze_layout      = MAZE_2,
    )

    renderer.set_banner("Act 4b — Retrained on MAZE_2  |  now it knows the spiral path")
    renderer.demo_play(agent2, n_episodes=3, fps=5)

    return renderer.wait_for_space(
        "Act 5 — Hyperparameter effect",
        "γ = 0.1  (shortsighted)  vs  γ = 0.95  (standard)   →   SPACE"
    )


def act5_hyperparams(renderer):
    """Compare γ=0.1 vs γ=0.95 visually."""
    _header(5, "Hyperparameter effect — discount factor γ")
    print("  γ controls how far ahead the agent plans.")
    print("  γ=0.1: only cares about the next 2-3 steps → can't value a goal 16 steps away.")
    print("  γ=0.95: plans far ahead → converges cleanly.\n")

    # --- Bad: gamma=0.1 ---
    print("  [Bad config]  γ = 0.1  — training 300 episodes...")
    env_bad = MazeEnv(MAZE_1)
    renderer.env = env_bad
    renderer.set_banner(
        "Act 5a — γ = 0.1  (shortsighted)  |  goal 16 steps away → discounted to nearly zero"
    )

    agent_bad, log_bad = train(
        n_episodes       = 300,
        renderer         = renderer,
        snapshot_episodes= SNAPSHOTS_BAD,
        verbose          = True,
        maze_layout      = MAZE_1,
        gamma            = 0.1,
    )

    wr_bad = np.mean([e["won"] for e in log_bad[-100:]]) * 100
    print(f"\n  γ=0.1 final win rate: {wr_bad:.0f}%  ← watch this number")

    if not renderer.wait_for_space(
        f"γ=0.1 win rate: {wr_bad:.0f}%   →   now try γ=0.95",
        "SPACE to train with good config"
    ):
        return

    # --- Good: gamma=0.95 ---
    print("\n  [Good config]  γ = 0.95  — training 300 episodes...")
    env_good = MazeEnv(MAZE_1)
    renderer.env = env_good
    renderer.set_banner(
        "Act 5b — γ = 0.95  (standard)  |  plans 16+ steps ahead → reaches the goal"
    )

    agent_good, log_good = train(
        n_episodes       = 300,
        renderer         = renderer,
        snapshot_episodes= SNAPSHOTS_MAZE1,
        verbose          = True,
        maze_layout      = MAZE_1,
        gamma            = 0.95,
    )

    wr_good = np.mean([e["won"] for e in log_good[-100:]]) * 100
    print(f"\n  γ=0.95 final win rate: {wr_good:.0f}%")
    print(f"\n  Summary — γ=0.1: {wr_bad:.0f}% win rate   vs   γ=0.95: {wr_good:.0f}% win rate")

    renderer.wait_for_space(
        f"γ=0.1: {wr_bad:.0f}% win rate    vs    γ=0.95: {wr_good:.0f}% win rate",
        "Demo complete — thank you!"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Pac-Man — presentation demo")
    parser.add_argument("--act",  type=int, default=0,
                        help="Jump to a specific act (1-5). Default: run all.")
    parser.add_argument("--fps",  type=int, default=0,
                        help="Override all fps values (useful for slower machines)")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL Pac-Man — Q-learning  |  Presentation Demo")
    print("=" * 60)
    print("\n  Controls during any episode:")
    print("    [Q]     policy arrows (best action per cell)")
    print("    [H]     Q-value heatmap (blue=low, red=high)")
    print("    [SPACE] pause / resume")
    print("    [ESC]   quit")
    print("\n  SPACE or ENTER to advance between acts.")
    print("  Starting pygame window...\n")

    # Scale fps overrides if --fps is set
    if args.fps > 0:
        for d in (SNAPSHOTS_MAZE1, SNAPSHOTS_MAZE2, SNAPSHOTS_BAD):
            for k in d:
                d[k] = min(d[k], args.fps)

    env      = MazeEnv(MAZE_1)
    renderer = PacManRenderer(env, fps=6)

    start_act = args.act if args.act in range(1, 6) else 1

    try:
        if start_act <= 1:
            if not act1_random(renderer):
                renderer.close(); return

        if start_act <= 2:
            agent = act2_training(renderer)
            if agent is None:
                renderer.close(); return
        else:
            # If jumping in mid-demo, train silently to get an agent
            print("  (Training silently to get a baseline agent...)")
            agent, _ = train(300, verbose=False, maze_layout=MAZE_1)
            renderer._agent = agent

        if start_act <= 3:
            if not act3_trained(renderer, agent):
                renderer.close(); return

        if start_act <= 4:
            if not act4_transfer(renderer, agent):
                renderer.close(); return

        if start_act <= 5:
            act5_explore_vs_exploit(renderer, agent)

    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    finally:
        renderer.close()

    print("\nDemo finished. results/ has reward_curve.png and rewards.csv")


if __name__ == "__main__":
    main()
