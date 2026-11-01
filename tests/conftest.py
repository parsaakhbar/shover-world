import pytest
from environment import ShoverWorldEnv

@pytest.fixture
def env():
    """
    Creates a standard environment for testing.
    We use a small internal size (5x5) for easier coordinate calculation.
    The actual grid will be 7x7 due to the Lava Border.
    Playable area indices are [1,1] to [5,5].
    """
    return ShoverWorldEnv(
        render_mode=None,
        n_rows=5,
        n_cols=5,
        initial_stamina=1000.0,
        initial_force=40.0,
        unit_force=10.0,
        perf_sq_initial_age=10,
        map_path=None,  # Use generator (we will overwrite grid manually)
        seed=42
    )

@pytest.fixture
def clear_grid():
    """
    Returns a function to wipe the inner board to EMPTY.
    """
    def _clear(env):
        # Keep borders as Lava (-100)
        h, w = env.grid.shape
        env.grid[1:h-1, 1:w-1] = 0
        env.box_momentum_mask[:] = 0
        env.perfect_squares = []
        # Reset stamina for consistency
        env.stamina = env.initial_stamina_cap
    return _clear