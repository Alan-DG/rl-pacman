"""
maze.py — Pac-Man gridworld environments

The agent starts top-left (1,1) and must reach GOAL (8,8).

Two maze layouts are provided:
  MAZE_1 — branching paths, diagonal route through the middle
  MAZE_2 — tight spiral, completely different optimal path
             (used to demonstrate that Q-learning memorises, not generalises)

State  : (row, col)
Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT
"""

import copy

# ---------------------------------------------------------------------------
# Maze 1 — default training maze
# Optimal path: right along row 1, then zigzag through middle to (8,8)
# ---------------------------------------------------------------------------
MAZE_1 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# ---------------------------------------------------------------------------
# Maze 2 — spiral layout
# Optimal path: right along row 1 → down right side → left along row 3 →
#               down left side → right along row 5 → down right side → goal
# The agent trained on MAZE_1 will get stuck: it tries to go DOWN at (1,4)
# which is a wall in MAZE_2, and its Q-table has no knowledge of this layout.
# ---------------------------------------------------------------------------
MAZE_2 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Keep backward-compatible name
DEFAULT_MAZE = MAZE_1

AGENT_START = (1, 1)
GOAL        = (8, 8)

R_STEP    =  -1
R_GOAL    = 100
MAX_STEPS = 400


class MazeEnv:
    """
    Pac-Man gridworld — navigate from start to goal.

    State  : (row, col)
    Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT
    Win    : agent reaches GOAL
    """

    ACTIONS      = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, maze_layout=None):
        self.maze_layout = maze_layout if maze_layout is not None else MAZE_1
        self.nrows       = len(self.maze_layout)
        self.ncols       = len(self.maze_layout[0])
        self.reset()

    def reset(self):
        self.grid      = copy.deepcopy(self.maze_layout)
        self.agent_pos = list(AGENT_START)
        self.steps     = 0
        self.done      = False
        self.grid[AGENT_START[0]][AGENT_START[1]] = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")

        self.steps += 1
        dr, dc = self.ACTIONS[action]
        new_r  = self.agent_pos[0] + dr
        new_c  = self.agent_pos[1] + dc
        reward = R_STEP

        if not self._is_wall(new_r, new_c):
            self.agent_pos = [new_r, new_c]

        if tuple(self.agent_pos) == GOAL:
            reward   += R_GOAL
            self.done = True

        if self.steps >= MAX_STEPS:
            self.done = True

        info = {"steps": self.steps}
        return self._get_state(), reward, self.done, info

    def _get_state(self):
        return tuple(self.agent_pos)

    def _is_wall(self, r, c):
        if r < 0 or r >= self.nrows or c < 0 or c >= self.ncols:
            return True
        return self.maze_layout[r][c] == 1

    @property
    def action_space(self):
        return list(self.ACTIONS.keys())
