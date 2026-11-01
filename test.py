"""
Shover World - Smoke test (headless).

Runs one random episode with no rendering and prints the episode return.

Usage:
    python test.py
"""
import numpy as np
from environment import ShoverWorldEnv

env = ShoverWorldEnv(render_mode=None, n_rows=6, n_cols=9,
                     initial_force=4.0, unit_force=1.0, seed=0)
env.action_space.seed(0)
obs, info = env.reset()
done = False
total_r = 0.0
while not done:
    a = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(a)
    done = terminated or truncated
    total_r += r
print("Episode return:", total_r)
print("Steps:", info["timestep"], "| Stamina left:", info["stamina"],
      "| Boxes left:", info["number_of_boxes"])
env.close()
