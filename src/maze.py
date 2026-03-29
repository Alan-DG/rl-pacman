"""
maze.py — Pac-Man gridworld environment

The agent starts top-left and must reach the GOAL cell (bottom-right).
Pellets scattered through the maze give small bonus rewards along the way,
but the win condition is simply: reach the goal.

This keeps the state space small — state = (row, col) — so Q-learning
converges quickly and cleanly.

Grid cell values:
  0 = empty floor
  1 = wall
  2 = pellet  (+5 bonus, visual)
  3 = power pellet (+15 bonus, visual)
  G = goal cell (+100, episode ends)
"""

import copy

DEFAULT_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 2, 1, 2, 1, 2, 1],
    [1, 2, 1, 3, 2, 2, 2, 1, 2, 1],
    [1, 2, 2, 2, 1, 1, 2, 2, 2, 1],
    [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
    [1, 2, 2, 2, 1, 1, 2, 2, 2, 1],
    [1, 2, 1, 2, 2, 2, 2, 1, 2, 1],
    [1, 2, 1, 1, 2, 1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

AGENT_START = (1, 1)
GOAL        = (8, 8)   # agent must reach this cell to win

R_STEP         =  -1
R_PELLET       =   5
R_POWER_PELLET =  15
R_GOAL         = 100   # large reward for reaching the exit
MAX_STEPS      = 300


class MazeEnv:
    """
    Pac-Man gridworld — navigate to the goal, eat pellets along the way.

    State  : (row, col)      — 45 reachable cells → tiny Q-table
    Actions: 0=UP 1=DOWN 2=LEFT 3=RIGHT
    Win    : agent reaches GOAL cell
    """

    ACTIONS      = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, maze_layout=None):
        self.maze_layout   = maze_layout or DEFAULT_MAZE
        self.nrows         = len(self.maze_layout)
        self.ncols         = len(self.maze_layout[0])
        self.total_pellets = sum(
            1 for row in self.maze_layout for cell in row if cell in (2, 3)
        )
        self.reset()

    def reset(self):
        self.grid          = copy.deepcopy(self.maze_layout)
        self.agent_pos     = list(AGENT_START)
        self.pellets_eaten = 0
        self.steps         = 0
        self.done          = False
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

        r, c = self.agent_pos
        cell = self.grid[r][c]

        # Pellet bonuses (visual + small reward, not the win condition)
        if cell == 2:
            reward += R_PELLET
            self.pellets_eaten += 1
            self.grid[r][c] = 0
        elif cell == 3:
            reward += R_POWER_PELLET
            self.pellets_eaten += 1
            self.grid[r][c] = 0

        # Win: reached the goal
        if tuple(self.agent_pos) == GOAL:
            reward    += R_GOAL
            self.done  = True

        # Timeout
        if self.steps >= MAX_STEPS:
            self.done = True

        info = {
            "pellets_eaten": self.pellets_eaten,
            "total_pellets": self.total_pellets,
            "steps":         self.steps,
            "won":           tuple(self.agent_pos) == GOAL or
                             (self.done and tuple(self.agent_pos) == GOAL),
        }
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
