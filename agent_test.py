"""
Shover World - Random Agent Mode

Run this file to watch a Random Agent play.

Usage:
    1. With Graphics (Watch it play):
       python test_agent.py

    2. Headless (Fast, no window):
       python test_agent.py --headless
"""
import time
import numpy as np
import os
import argparse
import sys

try:
    from shover_world.environment import ShoverWorldEnv
except ImportError:
    from environment import ShoverWorldEnv


def run_random_agent(headless=False):
    # Setup path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, "maps", "random_map.txt")
    if not os.path.exists(map_path):
        map_path = None

    print(f"--- STARTING RANDOM AGENT (Headless: {headless}) ---")

    # Determine Render Mode
    mode = None if headless else 'human'

    env = ShoverWorldEnv(
        render_mode=mode,
        map_path=map_path,
        max_timestep=300
    )

    # Run 3 Episodes
    for episode in range(1, 4):
        obs, info = env.reset()
        term = False
        trunc = False

        print(f"\n--- Episode {episode} Start ---")

        while not (term or trunc):

            # --- 1. Handle GUI & Quit Events (Only in Human Mode) ---
            if not headless:
                # We must call render to keep the window alive
                env.render()

                # Check if user clicked 'X' or pressed 'Q'
                if env.gui:
                    user_signal, _ = env.gui.handle_input()
                    if user_signal == -1:
                        print(">> Quit signal received. Exiting...")
                        env.close()
                        return  # Exit the function immediately

            # --- 2. Random Action Selection ---
            # Pick a random coordinate within valid map bounds
            r = np.random.randint(0, env.n_rows)
            c = np.random.randint(0, env.n_cols)

            # Pick a random action (0-6)
            a = np.random.randint(0, 7)

            action = ((r, c), a)

            # --- 3. Step Environment ---
            obs, reward, term, trunc, info = env.step(action)

            # --- 4. Timing & logging ---
            if not headless:
                # Slow down visual playback
                time.sleep(0.05)

        # End of Episode Logging
        reason = "Time Limit" if trunc else "Terminated"
        print(
            f"Episode Finished ({reason}). Steps: {info['timestep']} | Stamina Left: {info['stamina']}")

        if not headless:
            time.sleep(1)

    env.close()
    print("Test Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Agent")
    parser.add_argument("--headless", action="store_true",
                        help="Run without graphics for faster testing")
    args = parser.parse_args()

    run_random_agent(headless=args.headless)
