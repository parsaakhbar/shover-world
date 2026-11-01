def test_stamina_push_cost(env, clear_grid):
    clear_grid(env)
    env.grid[2, 2] = env.BOX
    
    # Formula: Unit(10) * k(1) + Initial(40) = 50
    expected_cost = 10.0 * 1 + 40.0
    start_stamina = env.stamina
    
    env.step(((2, 2), 2))
    
    assert env.stamina == start_stamina - expected_cost

def test_lava_refund(env, clear_grid):
    clear_grid(env)
    # Box at (2, 1), Lava border is at (2, 0) because inner starts at 1
    env.grid[2, 1] = env.BOX
    
    # Push Left (4) into Lava
    # Cost: 10*1 + 40 = 50
    # Refund: 40
    # Net Loss: 10
    start_stamina = env.stamina
    env.step(((2, 1), 4))
    
    assert env.grid[2, 1] == env.EMPTY # Box gone
    assert env.stamina == start_stamina - 10.0

def test_barrier_maker_gain(env, clear_grid):
    clear_grid(env)
    # Create 2x2 square
    env.grid[1:3, 1:3] = env.BOX
    # No-op step to trigger detection logic
    env.step(((0,0), 0)) 
    
    # Action 5: Barrier Maker
    # Gain: unit_force * n*n = 10 * 4 = 40 stamina (spec sec. 14 default)
    start_stamina = env.stamina
    env.step(((1, 1), 5))
    
    # Current impl: Barrier Maker doesn't COST stamina to cast, only rewards?
    # Logic: It rewards +unit_force * n^2.
    assert env.stamina == start_stamina + 40.0

def test_hellify_effects(env, clear_grid):
    clear_grid(env)
    # Create 3x3 square (required for Hellify n>2)
    env.grid[1:4, 1:4] = env.BOX
    env.step(((0,0), 0)) # Detection step
    
    # Action 6: Hellify
    env.step(((1, 1), 6))
    
    # 3x3 Square at (1,1) to (3,3)
    # Center (2,2) should be LAVA
    # Border should be EMPTY
    assert env.grid[2, 2] == env.LAVA
    assert env.grid[1, 1] == env.EMPTY

def test_momentum_cost_reduction(env, clear_grid):
    clear_grid(env)
    env.grid[2, 2] = env.BOX
    
    start = env.stamina
    
    # Step 1: Push Right. Stationary start. Cost 50.
    env.step(((2, 2), 2))
    stamina_t1 = env.stamina
    assert stamina_t1 == start - 50.0
    
    # Box is now at (2, 3).
    # Step 2: Push Right again. Non-stationary. Cost 10.
    env.step(((2, 3), 2)) 
    stamina_t2 = env.stamina
    
    assert stamina_t2 == stamina_t1 - 10.0

# --- Stationary / non-stationary transitions (spec sec. 4.3) ---

def test_consecutive_push_same_direction_skips_initial_force(env, clear_grid):
    clear_grid(env)
    env.grid[2, 1] = env.BOX
    env.step(((2, 1), 2))                     # stationary: 40 + 10
    before = env.stamina
    env.step(((2, 2), 2))                     # non-stationary: 10 only
    assert env.stamina == before - env.unit_force


def test_idle_step_resets_to_stationary(env, clear_grid):
    clear_grid(env)
    env.grid[2, 1] = env.BOX
    env.step(((2, 1), 2))
    env.step(((0, 0), 0))                     # no-op: box did not move
    before = env.stamina
    env.step(((2, 2), 2))                     # stationary again
    assert env.stamina == before - (env.initial_force + env.unit_force)


def test_stationary_is_direction_specific(env, clear_grid):
    clear_grid(env)
    env.grid[2, 2] = env.BOX
    env.step(((2, 2), 3))                     # down; box now at (3, 2)
    before = env.stamina
    env.step(((3, 2), 3))                     # down again: non-stationary in down
    assert env.stamina == before - env.unit_force
    before = env.stamina
    env.step(((4, 2), 2))                     # right: still stationary in right
    assert env.stamina == before - (env.initial_force + env.unit_force)


def test_initial_force_flag_in_info(env, clear_grid):
    clear_grid(env)
    env.grid[2, 1] = env.BOX
    _, _, _, _, info = env.step(((2, 1), 2))
    assert info["initial_force_charged"] is True
    assert info["chain_length_k"] == 1
    _, _, _, _, info = env.step(((2, 2), 2))
    assert info["initial_force_charged"] is False


# --- Invalid actions and edges (spec sec. 10, item 5) ---

def test_select_lava_cell_is_invalid(env, clear_grid):
    clear_grid(env)
    before = env.stamina
    _, _, _, _, info = env.step(((0, 2), 3))  # border lava cell
    assert env.stamina == before
    assert info["last_action_valid"] is False
    assert env.grid[0, 2] == env.LAVA


def test_unavailable_special_changes_nothing(env, clear_grid):
    clear_grid(env)
    env.grid[2, 2] = env.BOX
    snapshot = env.grid.copy()
    before = env.stamina
    for action in (5, 6):
        _, _, _, _, info = env.step(((2, 2), action))
        assert info["last_action_valid"] is False
    assert (env.grid == snapshot).all()
    assert env.stamina == before


def test_push_box_at_edge_into_border_lava(env, clear_grid):
    clear_grid(env)
    env.grid[1, 2] = env.BOX                  # top row of the playable area
    before = env.stamina
    _, _, _, _, info = env.step(((1, 2), 1))  # push up into the lava border
    assert env.grid[1, 2] == env.EMPTY
    assert env.grid[0, 2] == env.LAVA
    assert info["lava_destroyed_this_step"] == 1
    # cost 40 + 10, refund 40
    assert env.stamina == before - env.unit_force
