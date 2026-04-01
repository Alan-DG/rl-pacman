"""
renderer.py — Pygame visual renderer

Controls — keyboard or click the on-screen buttons:
  Q / [Policy]   — toggle policy arrow overlay (best action per cell)
  H / [Heatmap]  — toggle Q-value heatmap (cell colour = learned value)
  SPACE / [Pause]— pause / resume
  S / [Skip]     — skip current episode (moves to next, or exits act)
  - / [−]        — decrease FPS
  + / [+]        — increase FPS
  ESC / [Quit]   — quit demo
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

# Button colours
BTN_BG         = ( 28,  28,  55)
BTN_BG_HOVER   = ( 45,  45,  90)
BTN_BG_ACTIVE  = (  0, 130,  80)
BTN_BORDER     = ( 80,  80, 180)
BTN_BORDER_ACT = (  0, 210, 110)
BTN_TEXT       = (210, 210, 255)
BTN_SKIP_BG    = ( 80,  30,  10)
BTN_SKIP_ACT   = (200,  70,  20)
BTN_QUIT_BG    = ( 70,  10,  10)
BTN_QUIT_ACT   = (200,  20,  20)
BTN_FPS_BG     = ( 20,  40,  70)

CELL   = 52
MARGIN = 10
HUD_H  = 110   # taller to fit stats row + button bar

# FPS steps available via +/- buttons
FPS_STEPS = [1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 30]

ARROW_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


# ---------------------------------------------------------------------------
# Tiny button helper
# ---------------------------------------------------------------------------

class _Button:
    """A labelled rectangle that can be clicked or toggled."""

    def __init__(self, rect, label, key_hint="",
                 toggle=False, danger=False, fps_btn=False):
        self.rect     = pygame.Rect(rect)
        self.label    = label
        self.key_hint = key_hint
        self.toggle   = toggle
        self.danger   = danger
        self.fps_btn  = fps_btn
        self.active   = False
        self.hovered  = False

    def draw(self, surface, font_sm, font_md):
        if self.danger:
            bg = BTN_SKIP_ACT if self.hovered else (BTN_QUIT_ACT if self.active else BTN_QUIT_BG)
            bd = (255, 80, 40)
        elif self.fps_btn:
            bg = BTN_BG_HOVER if self.hovered else BTN_FPS_BG
            bd = BTN_BORDER
        elif self.toggle and self.active:
            bg = BTN_BG_HOVER if self.hovered else BTN_BG_ACTIVE
            bd = BTN_BORDER_ACT
        else:
            bg = BTN_BG_HOVER if self.hovered else BTN_BG
            bd = BTN_BORDER

        pygame.draw.rect(surface, bg,  self.rect, border_radius=5)
        pygame.draw.rect(surface, bd,  self.rect, 1, border_radius=5)

        if self.key_hint:
            hint_surf = font_sm.render(self.key_hint, True, (120, 120, 160))
            surface.blit(hint_surf, (self.rect.x + 4, self.rect.y + 2))

        lbl_surf = font_md.render(self.label, True, BTN_TEXT)
        lx = self.rect.centerx - lbl_surf.get_width() // 2
        ly = self.rect.centery - lbl_surf.get_height() // 2 + 4
        surface.blit(lbl_surf, (lx, ly))

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def check_click(self, pos):
        return self.rect.collidepoint(pos)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class PacManRenderer:
    def __init__(self, env, cell_size=CELL, fps=8):
        if not PYGAME_AVAILABLE:
            raise ImportError("Install pygame:  pip install pygame")
        self.env      = env
        self.cell     = cell_size
        self.fps      = fps
        self._frame   = 0
        self._show_q  = False
        self._heatmap = False
        self._paused  = False
        self._skip    = False
        self._banner  = None
        self._agent   = None
        self._buttons = []
        self._init()

    def _init(self):
        pygame.init()
        w = self.env.ncols * self.cell + 2 * MARGIN
        h = self.env.nrows * self.cell + 2 * MARGIN + HUD_H
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("RL Pac-Man — Q-learning")
        self.clock = pygame.time.Clock()
        try:
            self.font_sm = pygame.font.SysFont("monospace", 11)
            self.font_md = pygame.font.SysFont("monospace", 13, bold=True)
            self.font_lg = pygame.font.SysFont("monospace", 24, bold=True)
        except Exception:
            self.font_sm = pygame.font.Font(None, 16)
            self.font_md = pygame.font.Font(None, 18)
            self.font_lg = pygame.font.Font(None, 30)
        self._build_buttons()

    def _build_buttons(self):
        w  = self.screen.get_width()
        h  = self.screen.get_height()
        y0 = h - HUD_H
        by  = y0 + HUD_H - 42
        bh  = 34
        pad = 6

        specs = [
            # label       hint     w    toggle danger fps
            ("Policy",  "[Q]",    64,  True,  False, False),
            ("Heatmap", "[H]",    72,  True,  False, False),
            ("Pause",   "[SPC]",  64,  True,  False, False),
            ("Skip",    "[S]",    54,  False, True,  False),
            ("-",       "",       32,  False, False, True),
            ("8fps",    "",       52,  False, False, True),   # FPS display
            ("+",       "",       32,  False, False, True),
            ("Quit",    "[ESC]",  60,  False, True,  False),
        ]

        x = MARGIN
        self._buttons = []
        for label, hint, bw, toggle, danger, fps_btn in specs:
            btn = _Button((x, by, bw, bh), label, hint, toggle, danger, fps_btn)
            self._buttons.append(btn)
            x += bw + pad

        self._btn_policy  = self._buttons[0]
        self._btn_heatmap = self._buttons[1]
        self._btn_pause   = self._buttons[2]
        self._btn_skip    = self._buttons[3]
        self._btn_fps_dn  = self._buttons[4]
        self._btn_fps_lbl = self._buttons[5]
        self._btn_fps_up  = self._buttons[6]
        self._btn_quit    = self._buttons[7]

    # ------------------------------------------------------------------
    # FPS helpers
    # ------------------------------------------------------------------

    def _fps_step(self):
        diffs = [abs(f - self.fps) for f in FPS_STEPS]
        return diffs.index(min(diffs))

    def _fps_decrease(self):
        i = self._fps_step()
        if i > 0:
            self.fps = FPS_STEPS[i - 1]

    def _fps_increase(self):
        i = self._fps_step()
        if i < len(FPS_STEPS) - 1:
            self.fps = FPS_STEPS[i + 1]

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def _toggle_policy(self):
        self._show_q = not self._show_q
        self._btn_policy.active = self._show_q

    def _toggle_heatmap(self):
        self._heatmap = not self._heatmap
        self._btn_heatmap.active = self._heatmap

    def _toggle_pause(self):
        self._paused = not self._paused
        self._btn_pause.active = self._paused

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_banner(self, text):
        self._banner = text

    def render(self, episode=0, total_reward=0, epsilon=0.0):
        """Draw one frame. Returns False if window closed or ESC pressed."""
        self.clock.tick(self.fps)   # throttle here — before reading events
        mouse_pos = pygame.mouse.get_pos()
        for btn in self._buttons:
            btn.check_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._btn_policy.check_click(mouse_pos):
                    self._toggle_policy()
                elif self._btn_heatmap.check_click(mouse_pos):
                    self._toggle_heatmap()
                elif self._btn_pause.check_click(mouse_pos):
                    self._toggle_pause()
                elif self._btn_skip.check_click(mouse_pos):
                    self._skip = True
                elif self._btn_fps_dn.check_click(mouse_pos):
                    self._fps_decrease()
                elif self._btn_fps_up.check_click(mouse_pos):
                    self._fps_increase()
                elif self._btn_quit.check_click(mouse_pos):
                    return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_q:
                    self._toggle_policy()
                if event.key == pygame.K_h:
                    self._toggle_heatmap()
                if event.key == pygame.K_SPACE:
                    self._toggle_pause()
                if event.key == pygame.K_s:
                    self._skip = True
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._fps_decrease()
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._fps_increase()

        self._draw(episode, total_reward, epsilon, self._paused)
        return True

    def demo_play(self, agent, n_episodes=3, fps=6):
        """
        Run agent visually for n_episodes.
        [S] / Skip button skips current episode and moves to the next.
        """
        self._agent = agent
        saved_eps   = agent.epsilon
        self.fps    = fps

        for ep in range(1, n_episodes + 1):
            state      = self.env.reset()
            done       = False
            total      = 0
            open_      = True
            self._skip = False   # reset per episode

            while not done and open_ and not self._skip:
                while self._paused and open_ and not self._skip:
                    open_ = self.render(ep, total, 0.0)

                if not open_ or self._skip:
                    break

                action = agent.choose_action(state)
                state, reward, done, info = self.env.step(action)
                total += reward
                open_ = self.render(ep, total, 0.0)

            if not open_:
                break
            # _skip just ends this episode; outer loop continues

        agent.epsilon = saved_eps

    def wait_for_space(self, title, subtitle=""):
        """Transition screen. Returns True on SPACE/ENTER, False on ESC/close."""
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

        self._draw_goal_marker()
        self._draw_agent()

        if (self._show_q or self._heatmap) and self._agent:
            self._draw_q_overlay()

        self._draw_banner_overlay()
        self._draw_hud(episode, total_reward, epsilon, paused)
        pygame.display.flip()

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
        from maze import GOAL
        t    = self._frame
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

    def _heat_color(self, t):
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
        q_vals = {}
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                if self.env.maze_layout[r][c] != 1:
                    s = (r, c)
                    if s in self._agent.q_table:
                        q_vals[s] = float(np.max(self._agent.q_table[s]))

        min_q = min(q_vals.values()) if q_vals else 0.0
        max_q = max(q_vals.values()) if q_vals else 1.0
        rng   = max(max_q - min_q, 100.0)

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
                        color = (15, 15, 45)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (25, 25, 50), rect, 1)

    def _draw_banner_overlay(self):
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

        # Stats row
        rc = REWARD_POS if total_reward >= 0 else REWARD_NEG
        for i, (txt, col) in enumerate([
            (f"Episode  {episode}", TEXT_CLR),
            (f"Reward   {total_reward:+.0f}", rc),
        ]):
            self.screen.blit(self.font_md.render(txt, True, col),
                             (MARGIN, y0 + 8 + i * 20))

        eps_txt  = f"ε = {epsilon:.3f}" if epsilon > 0.001 else "greedy  (ε ≈ 0)"
        eps_surf = self.font_md.render(eps_txt, True, TEXT_DIM)
        self.screen.blit(eps_surf, (w - MARGIN - eps_surf.get_width(), y0 + 8))

        steps_surf = self.font_sm.render(f"Steps: {self.env.steps}", True, TEXT_DIM)
        self.screen.blit(steps_surf, (MARGIN, y0 + 50))

        if paused:
            p_surf = self.font_sm.render("PAUSED", True, (255, 200, 60))
            self.screen.blit(p_surf, (w // 2 - p_surf.get_width() // 2, y0 + 50))

        # Update live FPS label and draw all buttons
        self._btn_fps_lbl.label = f"{self.fps}fps"
        for btn in self._buttons:
            btn.draw(self.screen, self.font_sm, self.font_md)
