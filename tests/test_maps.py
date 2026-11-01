import pytest
import os
import tempfile
import numpy as np

def test_map_loading_valid(env):
    # Create a temporary map file: 3x3 grid
    content = "0 0 0\n0 10 0\n0 0 0"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        env.map_path = tmp_path
        env._load_or_generate_map()
        
        # 3x3 input map -> +2 border = 5x5 total grid
        assert env.grid.shape == (5, 5)
        assert env.grid[2, 2] == env.BOX # Center
        assert env.grid[0, 0] == env.LAVA # Border
    finally:
        os.remove(tmp_path)

def test_map_loading_malformed_rows(env):
    # Row 1 has 3 cols, Row 2 has 1 col
    content = "0 0 0\n10\n0 0 0"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        env.map_path = tmp_path
        # The loader catches exceptions and falls back to the generator
        env._load_or_generate_map()
        
        # If fallback happened, we get a random map.
        # Default inner size in fixture is 5x5 -> Grid 7x7
        assert env.grid is not None
        assert env.grid.shape == (7, 7) 
    finally:
        os.remove(tmp_path)

def test_map_loading_invalid_tokens(env):
    # 'X' is invalid, should map to 0/Empty (default fallback)
    content = "0 X 0\n0 0 0"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        env.map_path = tmp_path
        env._load_or_generate_map()
        
        # Total grid 4x5 (2 rows input + 2 border = 4)
        # Inner (0,1) maps to Grid (1,2)
        assert env.grid[1, 2] == env.EMPTY
    finally:
        os.remove(tmp_path)

MAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")


def _load(env, content):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        env.map_path = path
        env._load_or_generate_map()
    finally:
        os.remove(path)


def test_map_boxes_on_edges(env):
    _load(env, "10 0 10\n0 0 0\n10 0 10")
    assert env.grid.shape == (5, 5)
    for r, c in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        assert env.grid[r, c] == env.BOX
    assert env.grid[0, 0] == env.LAVA and env.grid[4, 4] == env.LAVA


def test_map_symbolic_format(env):
    _load(env, ". B #\nL A .")
    assert env.grid[1, 2] == env.BOX
    assert env.grid[1, 3] == env.BARRIER
    assert env.grid[2, 1] == env.LAVA
    assert env.grid[2, 3] == env.EMPTY


def test_map_agent_start_is_recorded(env):
    _load(env, ". . .\n. A .\n. . .")
    assert tuple(env.cursor_pos) == (2, 2)


@pytest.mark.parametrize("name", ["map_small.txt", "agent_small.txt", "custom_map.txt"])
def test_shipped_maps_load(env, name):
    env.map_path = os.path.join(MAPS_DIR, name)
    env._load_or_generate_map()
    assert env.grid[0, 0] == env.LAVA
    assert (env.grid == env.BOX).any()
