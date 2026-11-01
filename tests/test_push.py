import numpy as np

def test_push_single_box(env, clear_grid):
    clear_grid(env)
    # Place box at (2, 2)
    env.grid[2, 2] = env.BOX
    
    # Action: Select (2, 2), Push Right (2)
    obs, _, _, _, _ = env.step(((2, 2), 2))
    
    assert env.grid[2, 2] == env.EMPTY
    assert env.grid[2, 3] == env.BOX
    # Cursor should follow the box
    assert np.array_equal(env.cursor_pos, [2, 3])

def test_push_chain(env, clear_grid):
    clear_grid(env)
    # Place chain of 2 boxes: (2,2) and (2,3)
    env.grid[2, 2] = env.BOX
    env.grid[2, 3] = env.BOX
    
    # Push Right from (2, 2)
    env.step(((2, 2), 2))
    
    assert env.grid[2, 2] == env.EMPTY
    assert env.grid[2, 3] == env.BOX
    assert env.grid[2, 4] == env.BOX

def test_push_blocked_by_barrier(env, clear_grid):
    clear_grid(env)
    # Box at (2,2), Barrier at (2,3)
    env.grid[2, 2] = env.BOX
    env.grid[2, 3] = env.BARRIER
    
    prev_stamina = env.stamina
    
    # Try Push Right
    env.step(((2, 2), 2))
    
    # Assert nothing moved
    assert env.grid[2, 2] == env.BOX
    assert env.grid[2, 3] == env.BARRIER
    # Assert no stamina cost for invalid move
    assert env.stamina == prev_stamina

def test_push_empty_cell(env, clear_grid):
    clear_grid(env)
    # (2,2) is Empty
    prev_stamina = env.stamina
    
    # Select Empty, Try Push
    env.step(((2, 2), 2))
    
    assert env.stamina == prev_stamina
    assert env.grid[2, 2] == env.EMPTY

def test_momentum_change_direction(env, clear_grid):
    """
    Tests that changing direction resets momentum (Stationary status).
    """
    clear_grid(env)
    env.grid[2, 2] = env.BOX
    
    # Step 1: Push Right (Stationary) -> Cost 50
    env.step(((2, 2), 2)) 
    
    # Box is now at (2, 3).
    # Step 2: Push Down (Change Direction). 
    # Should pay Initial Force again.
    stamina_before = env.stamina
    env.step(((2, 3), 3)) 
    
    # Logic: 50 cost expected again
    cost = env.initial_force + env.unit_force * 1
    assert env.stamina == stamina_before - cost