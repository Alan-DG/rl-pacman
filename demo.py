"""
demo.py — One-command presentation demo

This is the script you run live during the presentation.

It trains the agent from scratch (fast, ~30 seconds), then immediately
launches the pygame window showing the trained agent playing the maze.

Usage
-----
    python demo.py                   # train 1500 episodes, then play 5 demo runs
    python demo.py --episodes 500    # quick train (faster but less converged)
    python demo.py --episodes 3000   # longer train (more polished behaviour)
    python demo.py --fps 4           # slower playback (easier to follow)
    python demo.py --fps 10          # faster playback

What the audience sees
-----------------------
1. Terminal output: reward climbing episode by episode → learning is working
2. Pygame window: the trained Pac-Man navigating the maze intelligently
3. Press [Q] to overlay the learned policy (directional arrows on each cell)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze     import MazeEnv
from train    import train
from renderer import PacManRenderer


def main():
    parser = argparse.ArgumentParser(description="RL Pac-Man — live demo")
    parser.add_argument("--episodes", type=int, default=1500,
                        help="Training episodes (default: 1500, ~30s)")
    parser.add_argument("--fps",      type=int, default=6,
                        help="Demo playback speed in frames/sec (default: 6)")
    parser.add_argument("--demo-runs", type=int, default=5,
                        help="How many demo episodes to show (default: 5)")
    parser.add_argument("--render-training", action="store_true",
                        help="Show pygame during training too (slower)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  RL Pac-Man — Q-learning demo")
    print("=" * 60)
    print(f"\nTraining for {args.episodes} episodes...")
    print("Watch the reward climb as the agent learns!\n")

    agent, log = train(
        n_episodes=args.episodes,
        render=args.render_training,
        render_every=100,
        verbose=True,
    )

    # Quick summary
    last100 = log[-100:]
    import numpy as np
    avg_r   = np.mean([e["total_reward"] for e in last100])
    win_r   = np.mean([e["won"]          for e in last100]) * 100
    print(f"\nTraining complete!")
    print(f"  Last 100 episodes — avg reward: {avg_r:.1f}  |  win rate: {win_r:.0f}%")
    print(f"  Q-table size: {agent.states_visited()} states visited")
    print(f"\nLaunching visual demo... (close window or press ESC to exit)")
    print("  [Q] toggles policy arrow overlay")
    print("  [SPACE] pauses the demo")

    # ------------------------------------------------------------------
    # 2. Visual demo
    # ------------------------------------------------------------------
    env      = MazeEnv()
    renderer = PacManRenderer(env, fps=args.fps)
    renderer._agent = agent

    renderer.demo_play(agent, n_episodes=args.demo_runs, fps=args.fps)
    renderer.close()

    print("\nDemo finished. Check results/ for reward_curve.png")


if __name__ == "__main__":
    main()
