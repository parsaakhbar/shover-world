"""
Regenerates the README screenshots in docs/images/ (headless, no window needed).
Each image is a BEFORE / AFTER pair taken from the real game renderer.

Usage (from the repo root):
    python docs/make_screenshots.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pygame
from environment import ShoverWorldEnv

OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)
pygame.font.init()
TITLE = pygame.font.SysFont("Arial", 22, bold=True)
LABEL = pygame.font.SysFont("Arial", 18, bold=True)
BODY = pygame.font.SysFont("Arial", 16)


def make_env(rows=7, cols=9):
    env = ShoverWorldEnv(render_mode="rgb_array", n_rows=rows, n_cols=cols, seed=0,
                         number_of_boxes=0, number_of_barriers=0, number_of_lavas=0)
    env.reset()
    env.grid[1:-1, 1:-1] = env.EMPTY
    env.box_momentum_mask[:] = 0
    env.perfect_squares = []
    return env


def frame(env):
    arr = env.render()
    return pygame.surfarray.make_surface(arr.swapaxes(0, 1))


def arrow(surf, row, col, n_boxes, color=(255, 255, 255)):
    """Draw a right-pointing arrow over a chain starting at grid cell (row, col)."""
    cs, y0 = 40, 60                                   # GUI cell size and HUD height
    y = y0 + row * cs + cs // 2
    x1, x2 = col * cs + 6, (col + n_boxes) * cs + cs - 6
    pygame.draw.line(surf, (0, 0, 0), (x1, y), (x2, y), 7)
    pygame.draw.polygon(surf, (0, 0, 0), [(x2 + 8, y), (x2 - 12, y - 14), (x2 - 12, y + 14)])
    pygame.draw.line(surf, color, (x1, y), (x2, y), 3)
    pygame.draw.polygon(surf, color, [(x2 + 4, y), (x2 - 10, y - 10), (x2 - 10, y + 10)])


def compose(name, title, caption, panels):
    """panels: list of (label, surface)."""
    w, h = panels[0][1].get_size()
    n = len(panels)
    pad, top, bottom = 16, 66, 60
    sheet = pygame.Surface((w * n + pad * (n + 1), h + top + bottom))
    sheet.fill((245, 245, 245))
    sheet.blit(TITLE.render(title, True, (20, 20, 20)), (pad, 10))
    for i, (label, img) in enumerate(panels):
        x = pad + i * (w + pad)
        sheet.blit(LABEL.render(label, True, (90, 90, 90)), (x, top - 24))
        sheet.blit(img, (x, top))
    sheet.blit(BODY.render(caption, True, (40, 40, 40)), (pad, top + h + 14))
    pygame.image.save(sheet, os.path.join(OUT, name))
    print("saved", name)


def compose_grid(name, title, panels, cols=2):
    """panels: list of (label, note, surface), laid out in a grid."""
    w, h = panels[0][2].get_size()
    rows = (len(panels) + cols - 1) // cols
    pad, head, label_h, note_h = 16, 56, 26, 30
    cell_h = label_h + h + note_h + pad
    sheet = pygame.Surface((w * cols + pad * (cols + 1), head + rows * cell_h + pad - pad))
    sheet.fill((245, 245, 245))
    sheet.blit(TITLE.render(title, True, (20, 20, 20)), (pad, 12))
    for i, (label, note, img) in enumerate(panels):
        x = pad + (i % cols) * (w + pad)
        y = head + (i // cols) * cell_h
        sheet.blit(LABEL.render(label, True, (60, 60, 60)), (x, y))
        sheet.blit(img, (x, y + label_h))
        sheet.blit(BODY.render(note, True, (40, 40, 40)), (x, y + label_h + h + 6))
    pygame.image.save(sheet, os.path.join(OUT, name))
    print("saved", name)


# 1. Pushing a chain: select with the mouse, then push; second push is cheaper
env = make_env(rows=5, cols=7)
env.grid[3, 2:5] = env.BOX                       # chain of 3 boxes
env.step(((0, 0), 0))
env.cursor_pos[:] = (1, 6)                       # selector parked on an empty cell
p0 = frame(env)

env.cursor_pos[:] = (3, 2)                       # user clicked the first box
p1 = frame(env)
arrow(p1, 3, 2, 3)

s0 = env.stamina
env.step(((3, 2), 2))                            # push right (stationary chain)
p2 = frame(env)
c1 = s0 - env.stamina
s1 = env.stamina
env.step(((3, 3), 2))                            # push right again (already moving)
p3 = frame(env)
c2 = s1 - env.stamina
compose_grid("01_push.png", "Pushing a chain of 3 boxes", [
    ("1. Nothing selected", "Click a box with the mouse to select it.", p0),
    ("2. Box selected (red outline)", "Press D / right arrow to push right.", p1),
    (f"3. After push 1:  -{c1:.0f} stamina",
     f"{env.initial_force:.0f} static friction + 3 x {env.unit_force:.0f}.", p2),
    (f"4. After push 2:  -{c2:.0f} stamina",
     f"Already moving, so only 3 x {env.unit_force:.0f}.", p3),
])

# 2. Barrier Maker: 2x2 perfect square -> barriers, stamina gain
env = make_env(rows=5, cols=7)
env.grid[2:4, 2:4] = env.BOX
env.step(((0, 0), 0))
env.cursor_pos[:] = (2, 2)
before = frame(env)
s0 = env.stamina
env.step(((2, 2), 5))
compose_grid("02_barrier_maker.png", "Barrier Maker (key B)", [
    ("BEFORE", "A 2x2 perfect square (isolated block of boxes).", before),
    (f"AFTER:  +{env.stamina - s0:.0f} stamina", "The square became barriers (dark gray).", frame(env)),
])

# 3. Hellify: 3x3 perfect square -> outer ring erased, center becomes lava
env = make_env(rows=5, cols=7)
env.grid[2:5, 2:5] = env.BOX
env.step(((0, 0), 0))
env.cursor_pos[:] = (2, 2)
before = frame(env)
env.step(((2, 2), 6))
compose_grid("03_hellify.png", "Hellify (key H)", [
    ("BEFORE", "A 3x3 perfect square.", before),
    ("AFTER", "Outer 8 boxes gone; the center is lava (orange).", frame(env)),
])
env.close()
