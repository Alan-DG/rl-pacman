"""
renderer.py — Pygame visual renderer

Keyboard controls during any running episode:
  Q      — toggle policy arrow overlay (best action per cell)
  H      — toggle Q-value heatmap (cell colour = learned value)
  SPACE  — pause / resume
  ESC    — quit demo
"""

import math, os, sys
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Colours
BLACK      = (  0,   0,   0)
WALL_BLUE  = ( 30,  30, 200)
WALL_EDGE  = ( 60,  60, 255)
FLOOR      = ( 10,  10,  20)
FLOOR_LINE = ( 22,  22,  38)
PELLET_CLR = (255, 184,  82)
POWER_CLR  = (255, 255, 255)
AGENT_CLR  = (255, 235,   0)
GOAL_CLR   = (  0, 230, 100)
TEXT_CLR   = (255, 255, 255)
TEXT_DIM   = (130, 130, 150)
HUD_BG     = ( 10,  10,  22)
HUD_LINE   = ( 50,  50, 220)
ARROW_CLR  = (  0, 210, 110)
REWARD_POS = (255, 210,   0)
REWARD_NEG = (255,  80,  80)
BANNER_BG  = (  0,   0,  12, 200)   # RGBA for alpha surface

CELL   = 52
MARGIN = 10
HUD_H  = 80   # includes banner line

ARROW_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class PacManRenderer:
    def __init__(self, env, cell_size=CELL, fps=8):
        if not PYGAME_AVAILABLE:
            raise ImportError("Install pygame:  pip install pygame")
        self.env     = env
        self.cell    = cell_size
        self.fps     = fps
        self._frame  = 0
        self._show_q = False
        self._heatmap= False
        self._paused = False
        self._banner = None
        self._agent  = None
        self._init()

    def _init(self):
        pygame.init()
        w = self.env.ncols * self.cell + 2 * MARGIN
        h = self.env.nrows * self.cell + 2 * MARGIN + HUD_H
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("RL Pac-Man — Q-learning")
        self.clock  = pygame.time.Clock()
        try:
            self.font_sm = pygame.font.SysFont("monospace", 13)
            self.font_md = pygame.font.SysFont("monospace", 15, bold=True)
            self.font_lg = pygame.font.SysFont("monospace", 24, bold=True)
        except Exception:
            self.font_sm = pygame.font.Font(None, 18)
            self.font_md = pygame.font.Font(None, 20)
            self.font_lg = pygame.font.Font(None, 30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_banner(self, text):
        """Set the banner text shown at the top of the screen during play."""
        self._banner = text

    def render(self, episode=0, total_reward=0, epsilon=0.0):
        """Draw one frame. Returns False if window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_q:
                    self._show_q = not self._show_q
                if event.key == pygame.K_h:
                    self._heatmap = not self._heatmap
                if event.key == pygame.K_SPACE:
                    self._paused = not self._paused
        self._draw(episode, total_reward, epsilon, self._paused)
        self.clock.tick(self.fps)
        return True

    def demo_play(self, agent, n_episodes=3, fps=6):
        """Run the trained agent visually for n_episodes."""
        self._agent = agent
        saved_eps   = agent.epsilon
        self.fps    = fps

        for ep in range(1, n_episodes + 1):
            state = self.env.reset()
            done  = False
            total = 0
            open_ = True

            while not done and open_:
                # While paused, keep drawing but don't step
                while self._paused and open_:
                    open_ = self.render(ep, total, 0.0)

                if not open_:
                    break

                action = agent.choose_action(state)
                state, reward, done, info = self.env.step(action)
                total += reward
                open_ = self.render(ep, total, 0.0)

            if not open_:
                break

        agent.epsilon = saved_eps

    def wait_for_space(self, title, subtitle=""):
        """
        Show a transition screen. Blocks until SPACE/ENTER (→ True)
        or ESC/close (→ False).
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        return True

            w, h = self.screen.get_size()
            self.screen.fill((5, 5, 15))

            t_surf = self.font_lg.render(title, True, (255, 255, 255))
            self.screen.blit(t_surf, (w // 2 - t_surf.get_width() // 2, h // 2 - 50))

            if subtitle:
                s_surf = self.font_md.render(subtitle, True, (160, 160, 200))
                self.screen.blit(s_surf, (w // 2 - s_surf.get_width() // 2, h // 2 + 5))

            hint = self.font_sm.render(
                "SPACE / ENTER to continue   ESC to quit", True, (70, 70, 100)
            )
            self.screen.blit(hint, (w // 2 - hint.get_width() // 2, h - 28))
            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        if PYGAME_AVAILABLE:
            pygame.quit()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, episode, total_reward, epsilon, paused=False):
        self._frame += 1
        self.screen.fill(BLACK)

        if self._heatmap and self._agent:
            self._draw_heatmap_grid()
        else:
            self._draw_grid()
            self._draw_pellets()

        self._draw_goal_marker()
        self._draw_agent()

        if (self._show_q or self._heatmap) and self._agent:
            self._draw_q_overlay()

        self._draw_banner_overlay()
        self._draw_hud(episode, total_reward, epsilon, paused)
        pygame.display.flip()

    # --- grid & tiles ---

    def _rect(self, r, c):
        return pygame.Rect(MARGIN + c * self.cell, MARGIN + r * self.cell,
                           self.cell, self.cell)

    def _center(self, r, c):
        rect = self._rect(r, c)
        return rect.centerx, rect.centery

    def _draw_grid(self):
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                rect = self._rect(r, c)
                if self.env.maze_layout[r][c] == 1:
                    pygame.draw.rect(self.screen, WALL_BLUE, rect)
                    pygame.draw.rect(self.screen, WALL_EDGE,  rect, 2)
                else:
                    pygame.draw.rect(self.screen, FLOOR,      rect)
                    pygame.draw.rect(self.screen, FLOOR_LINE, rect, 1)

    def _draw_pellets(self):
        t = self._frame
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                cell = self.env.grid[r][c]
                cx, cy = self._center(r, c)
                if cell == 2:
                    pygame.draw.circle(self.screen, PELLET_CLR, (cx, cy), 4)
                elif cell == 3:
                    pulse = 7 + int(3 * math.sin(t * 0.15))
                    pygame.draw.circle(self.screen, POWER_CLR,  (cx, cy), pulse)
                    pygame.draw.circle(self.screen, PELLET_CLR, (cx, cy), pulse - 2)

    def _draw_goal_marker(self):
        """Pulsing green marker on the goal cell."""
        from maze import GOAL
        t  = self._frame
        r, c = GOAL
        cx, cy = self._center(r, c)
        pulse  = 10 + int(4 * math.sin(t * 0.12))
        pygame.draw.circle(self.screen, GOAL_CLR, (cx, cy), pulse, 3)
        g_surf = self.font_sm.render("G", True, GOAL_CLR)
        self.screen.blit(g_surf, (cx - g_surf.get_width() // 2,
                                  cy - g_surf.get_height() // 2))

    def _draw_agent(self):
        r, c   = self.env.agent_pos
        cx, cy = self._center(r, c)
        radius = self.cell // 2 - 4
        t      = self._frame
        mouth  = 10 + int(20 * abs(math.sin(t * 0.25)))
        start  = math.radians(mouth)
        end    = math.radians(360 - mouth)
        pts    = [(cx, cy)]
        for i in range(41):
            a = start + (end - start) * i / 40
            pts.append((cx + radius * math.cos(a),
                        cy - radius * math.sin(a)))
        if len(pts) >= 3:
            pygame.draw.polygon(self.screen, AGENT_CLR, pts)
        pygame.draw.circle(
            self.screen, BLACK,
            (cx + int(radius * 0.35), cy - int(radius * 0.55)), 3
        )

    # --- Q-value overlays ---

    def _draw_q_overlay(self):
        """Directional arrows showing best learned action per cell."""
        policy = self._agent.get_policy_grid(self.env.nrows, self.env.ncols)
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                if self.env.maze_layout[r][c] == 1:
                    continue
                action = policy[r][c]
                if action is None:
                    continue
                cx, cy = self._center(r, c)
                surf   = self.font_sm.render(ARROW_SYMBOLS[action], True, ARROW_CLR)
                self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    def _heat_color(self, t):
        """Map t ∈ [0,1] → colour: dark blue → cyan → yellow → red."""
        stops = [
            (0.00, (10,  10,  70)),
            (0.30, ( 0, 130, 210)),
            (0.60, (220, 200,  10)),
            (1.00, (220,  40,  10)),
        ]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t <= t1:
                s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return tuple(int(c0[j] + s * (c1[j] - c0[j])) for j in range(3))
        return stops[-1][1]

    def _draw_heatmap_grid(self):
        """Colour each floor cell by its max Q-value (blue=low, red=high)."""
        q_vals = {}
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                if self.env.maze_layout[r][c] != 1:
                    s = (r, c)
                    if s in self._agent.q_table:
                        q_vals[s] = float(np.max(self._agent.q_table[s]))

        min_q = min(q_vals.values()) if q_vals else 0.0
        max_q = max(q_vals.values()) if q_vals else 1.0
        rng   = max(max_q - min_q, 1.0)

        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                rect = self._rect(r, c)
                if self.env.maze_layout[r][c] == 1:
                    pygame.draw.rect(self.screen, WALL_BLUE, rect)
                    pygame.draw.rect(self.screen, WALL_EDGE,  rect, 2)
                else:
                    s = (r, c)
                    if s in q_vals:
                        t     = (q_vals[s] - min_q) / rng
                        color = self._heat_color(t)
                    else:
                        color = (15, 15, 45)   # unvisited — near-black
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (25, 25, 50), rect, 1)

    # --- overlays ---

    def _draw_banner_overlay(self):
        """Semi-transparent banner at top of screen showing current act."""
        if not self._banner:
            return
        w = self.screen.get_width()
        surf = pygame.Surface((w, 20), pygame.SRCALPHA)
        surf.fill((0, 0, 10, 200))
        self.screen.blit(surf, (0, 0))
        text = self.font_sm.render(self._banner, True, (255, 220, 60))
        self.screen.blit(text, (8, 3))

    def _draw_hud(self, episode, total_reward, epsilon, paused):
        w  = self.screen.get_width()
        h  = self.screen.get_height()
        y0 = h - HUD_H
        pygame.draw.rect(self.screen, HUD_BG, pygame.Rect(0, y0, w, HUD_H))
        pygame.draw.line(self.screen, HUD_LINE, (0, y0), (w, y0), 1)

        pellets_left  = sum(1 for row in self.env.grid for c in row if c in (2, 3))
        pellets_eaten = self.env.total_pellets - pellets_left

        rc = REWARD_POS if total_reward >= 0 else REWARD_NEG
        for i, (txt, col) in enumerate([
            (f"Episode  {episode}", TEXT_CLR),
            (f"Reward   {total_reward:+.0f}", rc),
        ]):
            self.screen.blit(self.font_md.render(txt, True, col),
                             (MARGIN, y0 + 10 + i * 22))

        pellet_surf = self.font_md.render(
            f"Pellets  {pellets_eaten}/{self.env.total_pellets}", True, PELLET_CLR
        )
        self.screen.blit(pellet_surf,
                         (w // 2 - pellet_surf.get_width() // 2, y0 + 10))

        eps_txt  = f"ε = {epsilon:.3f}" if epsilon > 0.001 else "greedy  (ε ≈ 0)"
        eps_surf = self.font_md.render(eps_txt, True, TEXT_DIM)
        self.screen.blit(eps_surf,
                         (w - MARGIN - eps_surf.get_width(), y0 + 10))

        hint = ("PAUSED — SPACE to resume"
                if paused
                else "[Q] policy  [H] heatmap  [SPACE] pause  [ESC] quit")
        hint_surf = self.font_sm.render(hint, True, TEXT_DIM)
        self.screen.blit(hint_surf,
                         (w - MARGIN - hint_surf.get_width(), y0 + 56))

        steps_surf = self.font_sm.render(f"Steps: {self.env.steps}", True, TEXT_DIM)
        self.screen.blit(steps_surf, (MARGIN, y0 + 56))
