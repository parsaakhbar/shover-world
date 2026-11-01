"""
Shover World - Human Player Mode
Run this file to play the game manually with Keyboard and Mouse.
"""
import time
import numpy as np
import os
import sys

# Import Environment
try:
    from shover_world.environment import ShoverWorldEnv
except ImportError:
    from environment import ShoverWorldEnv


def get_map_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, "maps", filename)
    if not os.path.exists(map_path):
        return None
    return map_path


def run_human():
    # 1. Try to load a map, or fallback to generator
    map_file = get_map_path("custom_map.txt")
    use_gen = False

    if not map_file:
        print("(!) 'custom_map.txt' not found. Using Procedural Generator.")
        use_gen = True

    print(f"--- STARTING HUMAN MODE ---")
    if not use_gen:
        print(f"Map: {map_file}")

    # 2. Initialize Environment
    env = ShoverWorldEnv(
        render_mode='human',
        map_path=map_file if not use_gen else None,
        max_timestep=600,
        # Generator settings (ignored if map_path is set)
        n_rows=9, n_cols=13,
        number_of_boxes=20,
        number_of_barriers=5,
        number_of_lavas=3
    )

    obs, info = env.reset()
    running = True

    # Local cursor tracker
    current_cursor = env.cursor_pos

    print("\nControls:")
    print(" [Mouse] Click Box : Select target")
    print(" [WASD / Arrows] : Push Box / Move Cursor")
    print(" [B]             : Barrier Maker (Needs 2x2 Square)")
    print(" [H]             : Hellify (Needs 3x3 Square)")
    print(" [R]             : Reset Level")
    print(" [Q]             : Quit\n")

    while running:
        # Render GUI
        if env.gui is None:
            env.render()

        # Input Handling
        action_type, mouse_pos = env.gui.handle_input()

        if action_type == -1:  # Quit
            running = False
            continue

        if mouse_pos:
            current_cursor = np.array(mouse_pos, dtype=np.int32)
            env.cursor_pos = current_cursor

        if action_type == 99:  # Reset
            print("Resetting Level...")
            env.reset()
            current_cursor = env.cursor_pos
            continue

        # Execute Action
        if action_type != 0:
            action = (tuple(current_cursor), action_type)
            obs, reward, term, trunc, info = env.step(action)

            # Update local cursor (it might have moved with the box)
            current_cursor = env.cursor_pos

            if term or trunc:
                status = "Time Up" if trunc else "Terminated"
                print(
                    f">> Game Over: {status} | Final Stamina: {info['stamina']}")
                env.render()
                time.sleep(1.0)
                print("Replaying same map...")
                env.reset(options={'keep_map': True})

        env.render()
        # 30 FPS cap
        time.sleep(0.033)

    env.close()


if __name__ == "__main__":
    run_human()
