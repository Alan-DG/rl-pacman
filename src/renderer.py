"""
renderer.py — Pygame visual renderer for the RL Pac-Man demo

Controls during demo_play():
  SPACE  — pause / resume
  Q      — toggle Q-policy arrow overlay
  ESC    — quit
"""

import math
import os
import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Colours
BLACK       = (  0,   0,   0)
WALL_BLUE   = ( 30,  30, 200)
WALL_EDGE   = ( 60,  60, 255)
FLOOR       = ( 10,  10,  20)
FLOOR_LINE  = ( 22,  22,  38)
PELLET_CLR  = (255, 184,  82)
POWER_CLR   = (255, 255, 255)
AGENT_CLR   = (255, 235,   0)
TEXT_CLR    = (255, 255, 255)
TEXT_DIM    = (140, 140, 160)
HUD_BG      = ( 12,  12,  22)
HUD_LINE    = ( 50,  50, 220)
ARROW_CLR   = (  0, 210, 110)
REWARD_POS  = (255, 210,   0)
REWARD_NEG  = (255,  80,  80)

CELL   = 52
MARGIN = 10
HUD_H  = 68

ARROW_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class PacManRenderer:
    def __init__(self, env, cell_size=CELL, fps=8):
        if not PYGAME_AVAILABLE:
            raise ImportError("Install pygame:  pip install pygame")
        self.env    = env
        self.cell   = cell_size
        self.fps    = fps
        self._frame = 0
        self._show_q = False
        self._paused = False
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
            self.font_md = pygame.font.SysFont("monospace", 16, bold=True)
        except Exception:
            self.font_sm = pygame.font.Font(None, 18)
            self.font_md = pygame.font.Font(None, 22)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        self._draw(episode, total_reward, epsilon)
        self.clock.tick(self.fps)
        return True

    def demo_play(self, agent, n_episodes=5, fps=6):
        """Run the trained agent visually. Blocks until done or closed."""
        self._agent = agent
        self.fps    = fps
        saved_eps   = agent.epsilon
        agent.epsilon = 0.0

        for ep in range(1, n_episodes + 1):
            state = self.env.reset()
            done  = False
            total = 0
            open_ = True

            while not done and open_:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        open_ = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            open_ = False
                        if event.key == pygame.K_SPACE:
                            self._paused = not self._paused
                        if event.key == pygame.K_q:
                            self._show_q = not self._show_q

                if self._paused:
                    self._draw(ep, total, 0.0, paused=True)
                    self.clock.tick(10)
                    continue

                action = agent.choose_action(state)
                state, reward, done, info = self.env.step(action)
                total += reward
                self._draw(ep, total, 0.0)
                self.clock.tick(self.fps)

            if not open_:
                break

        agent.epsilon = saved_eps

    def close(self):
        if PYGAME_AVAILABLE:
            pygame.quit()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, episode, total_reward, epsilon, paused=False):
        self._frame += 1
        self.screen.fill(BLACK)
        self._draw_grid()
        self._draw_pellets()
        self._draw_agent()
        if self._show_q and self._agent:
            self._draw_q_overlay()
        self._draw_hud(episode, total_reward, epsilon, paused)
        pygame.display.flip()

    def _rect(self, r, c):
        return pygame.Rect(
            MARGIN + c * self.cell,
            MARGIN + r * self.cell,
            self.cell, self.cell
        )

    def _center(self, r, c):
        rect = self._rect(r, c)
        return rect.centerx, rect.centery

    def _draw_grid(self):
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                rect = self._rect(r, c)
                if self.env.maze_layout[r][c] == 1:
                    pygame.draw.rect(self.screen, WALL_BLUE,  rect)
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
            a  = start + (end - start) * i / 40
            pts.append((cx + radius * math.cos(a),
                        cy - radius * math.sin(a)))
        if len(pts) >= 3:
            pygame.draw.polygon(self.screen, AGENT_CLR, pts)
        # eye
        pygame.draw.circle(
            self.screen, BLACK,
            (cx + int(radius * 0.35), cy - int(radius * 0.55)), 3
        )

    def _draw_q_overlay(self):
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

    def _draw_hud(self, episode, total_reward, epsilon, paused):
        w  = self.screen.get_width()
        h  = self.screen.get_height()
        y0 = h - HUD_H
        pygame.draw.rect(self.screen, HUD_BG, pygame.Rect(0, y0, w, HUD_H))
        pygame.draw.line(self.screen, HUD_LINE, (0, y0), (w, y0), 1)

        pellets_left  = sum(1 for row in self.env.grid for cell in row if cell in (2, 3))
        pellets_eaten = self.env.total_pellets - pellets_left

        rc = REWARD_POS if total_reward >= 0 else REWARD_NEG
        for i, (txt, col) in enumerate([
            (f"Episode  {episode}", TEXT_CLR),
            (f"Reward   {total_reward:+.0f}", rc),
        ]):
            self.screen.blit(
                self.font_md.render(txt, True, col),
                (MARGIN, y0 + 10 + i * 24)
            )

        pellet_surf = self.font_md.render(
            f"Pellets  {pellets_eaten}/{self.env.total_pellets}", True, PELLET_CLR
        )
        self.screen.blit(pellet_surf, (w // 2 - pellet_surf.get_width() // 2, y0 + 10))

        eps_txt  = f"ε = {epsilon:.3f}" if epsilon > 0 else "greedy (ε = 0)"
        eps_surf = self.font_md.render(eps_txt, True, TEXT_DIM)
        self.screen.blit(eps_surf, (w - MARGIN - eps_surf.get_width(), y0 + 10))

        hint = "PAUSED — SPACE to resume" if paused else "[Q] policy  [SPACE] pause  [ESC] quit"
        hint_surf = self.font_sm.render(hint, True, TEXT_DIM)
        self.screen.blit(hint_surf, (w - MARGIN - hint_surf.get_width(), y0 + 38))

        steps_surf = self.font_sm.render(f"Steps: {self.env.steps}", True, TEXT_DIM)
        self.screen.blit(steps_surf, (MARGIN, y0 + 48))
