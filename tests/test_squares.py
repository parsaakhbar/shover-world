import numpy as np

def test_detect_2x2_square(env, clear_grid):
    clear_grid(env)
    env.grid[1:3, 1:3] = env.BOX
    
    squares = env._detect_perfect_squares()
    assert len(squares) == 1
    assert squares[0]['n'] == 2
    assert squares[0]['top_left'] == (1, 1)

def test_detect_3x3_square(env, clear_grid):
    clear_grid(env)
    env.grid[1:4, 1:4] = env.BOX
    
    squares = env._detect_perfect_squares()
    assert len(squares) == 1
    assert squares[0]['n'] == 3

def test_adjacency_breaks_square(env, clear_grid):
    clear_grid(env)
    # Create 2x2
    env.grid[1:3, 1:3] = env.BOX
    
    # Add an adjacent box touching the side
    env.grid[1, 3] = env.BOX 
    
    squares = env._detect_perfect_squares()
    assert len(squares) == 0 # Should not be isolated

def test_corner_adjacency_strictness(env, clear_grid):
    """
    Test that even diagonal touching breaks 'Perfect Square' isolation 
    (based on the implementation using a bounding box patch).
    """
    clear_grid(env)
    env.grid[1:3, 1:3] = env.BOX
    
    # Add box at diagonal (0, 0) - strictly outside the 2x2, but in the detection patch.
    env.grid[0, 0] = env.BOX
    
    squares = env._detect_perfect_squares()
    assert len(squares) == 0

def test_unavailable_special_action(env, clear_grid):
    clear_grid(env)
    # No boxes
    prev_stamina = env.stamina
    
    # Try Barrier Maker (5) without a square
    env.step(((1, 1), 5))
    
    # Assert no stamina change and no grid change (Barrier shouldn't appear)
    assert env.stamina == prev_stamina
    assert env.grid[1, 1] == env.EMPTY