"""
Shover World - Map Generator Utility

This script generates procedural grid maps for the Shover World environment.
It creates text files containing symbols representing the game state (Agent, Boxes, Barriers, Lava).

Modes of Operation:
1. Interactive Mode: Run without arguments. The script prompts for parameters.
   $ python map_generator.py

2. CLI Mode: Run with arguments. Missing arguments use default values.
   $ python map_generator.py --rows 15 --cols 15 --boxes 30 --filename large_map.txt
"""

import os
import sys
import random
import argparse

# --- Configuration Constants ---

# Target directory for saving generated maps.
# Always the folder containing this script, regardless of working directory.
MAPS_DIR = os.path.dirname(os.path.abspath(__file__))

# Default values used if the user just presses Enter or omits CLI flags.
DEFAULT_CONFIG = {
    "filename": "custom_map.txt",
    "rows": 9,
    "cols": 13,
    "boxes": 15,
    "barriers": 5,
    "lavas": 3
}

# Mapping of logical elements to character symbols in the text file.
SYMBOLS = {
    'EMPTY': '.',
    'BOX': 'B',
    'BARRIER': '#',
    'LAVA': 'L',
    'AGENT': 'A'
}


def ensure_maps_dir():
    """
    Checks if the target maps directory exists; creates it if it does not.
    """
    if not os.path.exists(MAPS_DIR):
        os.makedirs(MAPS_DIR)
        print(f"Created directory: {MAPS_DIR}")


def ask_input(prompt, default, value_type=str):
    """
    Helper function to prompt the user for input via the console.

    Args:
        prompt (str): The question to ask the user.
        default (any): The value to use if the user inputs nothing (presses Enter).
        value_type (type): The expected type (int, str, etc.) to cast the input to.

    Returns:
        The user's input cast to `value_type`, or the `default` value.
    """
    # Display prompt with default value in brackets
    user_input = input(f"{prompt} [{default}]: ").strip()

    # If user pressed Enter, return default
    if not user_input:
        return default

    try:
        return value_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default


def generate_grid(rows, cols, n_boxes, n_barriers, n_lavas):
    """
    Generates a random 2D grid of characters.

    Logic:
        1. Calculates total cells vs. requested elements to prevent overflow.
        2. Creates a list of all possible (row, col) coordinates.
        3. Shuffles the coordinates to ensure random placement without collision.
        4. Pops coordinates off the list to place the Agent, Boxes, Barriers, and Lava.

    Args:
        rows (int): Grid height.
        cols (int): Grid width.
        n_boxes (int): Number of boxes.
        n_barriers (int): Number of barriers.
        n_lavas (int): Number of lava cells.

    Returns:
        list[list[str]]: A 2D list representing the map grid.
    """
    total_cells = rows * cols
    # Calculate required space (including 1 spot for the Agent)
    needed = 1 + n_boxes + n_barriers + n_lavas

    # Validation: Ensure elements fit in the grid
    if needed > total_cells:
        print(f"(!) Error: Map size ({rows}x{cols}={total_cells}) is too small "
              f"for the requested elements ({needed}).")
        sys.exit(1)

    # Initialize empty grid filled with '.'
    grid = [[SYMBOLS['EMPTY'] for _ in range(cols)] for _ in range(rows)]

    # Create a "Deck" of all coordinates (0,0) to (rows-1, cols-1)
    coords = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(coords)  # Randomize position order

    # 1. Place Agent (Always 1)
    r, c = coords.pop()
    grid[r][c] = SYMBOLS['AGENT']

    # 2. Place Boxes
    for _ in range(n_boxes):
        r, c = coords.pop()
        grid[r][c] = SYMBOLS['BOX']

    # 3. Place Barriers
    for _ in range(n_barriers):
        r, c = coords.pop()
        grid[r][c] = SYMBOLS['BARRIER']

    # 4. Place Lava
    for _ in range(n_lavas):
        r, c = coords.pop()
        grid[r][c] = SYMBOLS['LAVA']

    return grid


def save_map(grid, filename):
    """
    Writes the generated grid to a text file.

    Args:
        grid (list[list[str]]): The 2D grid data.
        filename (str): The name of the file (e.g., 'map01.txt').
    """
    ensure_maps_dir()
    filepath = os.path.join(MAPS_DIR, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in grid:
                # Join row list into a space-separated string (e.g., ". . B . #")
                f.write(" ".join(row) + "\n")

        # Print success message and preview
        print(f"\n[Success] Map saved to: {filepath}")
        print("-" * 30)
        for row in grid:
            print(" ".join(row))
        print("-" * 30)

    except IOError as e:
        print(f"[Error] Could not save file: {e}")


def main():
    """
    Main execution flow. 
    Parses arguments to determine whether to run in Interactive or CLI mode,
    then triggers generation and saving.
    """
    parser = argparse.ArgumentParser(description="Shover World Map Generator")

    # Define optional command-line arguments
    parser.add_argument("--filename", type=str,
                        help="Output filename (e.g. my_map.txt)")
    parser.add_argument("--rows", type=int, help="Number of rows")
    parser.add_argument("--cols", type=int, help="Number of columns")
    parser.add_argument("--boxes", type=int, help="Number of boxes")
    parser.add_argument("--barriers", type=int, help="Number of barriers")
    parser.add_argument("--lavas", type=int,
                        help="Number of internal lava cells")

    args = parser.parse_args()

    # Logic: Check if ANY command-line argument was actually provided.
    # If yes -> CLI Mode. If no -> Interactive Mode.
    is_cli_mode = any(getattr(args, arg) is not None for arg in vars(args))

    config = DEFAULT_CONFIG.copy()

    if is_cli_mode:
        print(">> Mode: Command Line (Using provided args + defaults)")
        # Override defaults with any provided CLI arguments
        if args.filename:
            config["filename"] = args.filename
        if args.rows:
            config["rows"] = args.rows
        if args.cols:
            config["cols"] = args.cols
        if args.boxes:
            config["boxes"] = args.boxes
        if args.barriers:
            config["barriers"] = args.barriers
        if args.lavas:
            config["lavas"] = args.lavas

    else:
        print(">> Mode: Interactive (Press Enter to accept defaults)")
        # Ask user for each parameter step-by-step
        config["filename"] = ask_input(
            "Map Filename", DEFAULT_CONFIG["filename"])
        config["rows"] = ask_input("Rows", DEFAULT_CONFIG["rows"], int)
        config["cols"] = ask_input("Cols", DEFAULT_CONFIG["cols"], int)
        config["boxes"] = ask_input(
            "Number of Boxes", DEFAULT_CONFIG["boxes"], int)
        config["barriers"] = ask_input(
            "Number of Barriers", DEFAULT_CONFIG["barriers"], int)
        config["lavas"] = ask_input(
            "Number of Lava cells", DEFAULT_CONFIG["lavas"], int)

    # Log generation start
    print(
        f"\nGenerating map: {config['rows']}x{config['cols']} with {config['boxes']} boxes...")

    # Generate the data structure
    grid = generate_grid(
        config["rows"],
        config["cols"],
        config["boxes"],
        config["barriers"],
        config["lavas"]
    )

    # Save to file
    save_map(grid, config["filename"])


if __name__ == "__main__":
    main()
