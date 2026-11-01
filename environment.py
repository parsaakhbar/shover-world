import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pygame


class ShoverWorldEnv(gym.Env):
    """
    A Gymnasium Environment for the 'Shover World' puzzle game.

    The environment consists of a grid world where an agent (represented by a cursor)
    must select and push chains of boxes. The goal is to manage stamina while clearing
    boxes or forming specific patterns ('Perfect Squares') to unleash special abilities.

    Key Mechanics:
    - **Stamina**: Consumed by pushing. Higher cost for starting a stationary chain (Static Friction).
    - **Lava**: Acts as a border and hazard. Pushing boxes into lava refunds stamina.
    - **Perfect Squares**: Isolated NxN blocks of boxes that age over time. If active,
      they enable special actions (Barrier Maker / Hellify) or dissolve if ignored.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode='human',
                 n_rows=13, n_cols=9,
                 max_timestep=400,
                 number_of_boxes=15,
                 number_of_barriers=5,
                 number_of_lavas=3,
                 initial_stamina=1000.0,
                 initial_force=40.0,
                 unit_force=10.0,
                 perf_sq_initial_age=10,
                 map_path=None,
                 seed=None):
        """
        Initialize the Shover World environment.

        Args:
            render_mode (str): 'human' for GUI, 'rgb_array' for pixel data, or None for headless.
            n_rows (int): Height of the inner playable area (excluding lava border).
            n_cols (int): Width of the inner playable area.
            max_timestep (int): Episode time limit.
            number_of_boxes (int): Number of boxes to place in procedurally generated maps.
            number_of_barriers (int): Number of barriers to place in procedurally generated maps.
            number_of_lavas (int): Number of internal lava cells in procedurally generated maps.
            initial_stamina (float): Starting stamina pool.
            initial_force (float): Stamina cost for starting to move a stationary box (Static Friction).
            unit_force (float): Stamina cost per box in a chain (Kinetic Friction).
            perf_sq_initial_age (int): Number of steps a Perfect Square exists before dissolving.
            map_path (str, optional): Path to a .txt map file. If None, a random map is generated.
            seed (int, optional): RNG seed for reproducibility.
        """

        super().__init__()

        self.render_mode = render_mode
        self.max_timestep = max_timestep
        self.initial_stamina_cap = initial_stamina
        self.initial_force = initial_force
        self.unit_force = unit_force
        self.perf_sq_limit = perf_sq_initial_age
        self.map_path = map_path

        # Map Generation Parameters
        self.inner_rows = n_rows
        self.inner_cols = n_cols
        self.n_boxes = number_of_boxes
        self.n_barriers = number_of_barriers
        self.n_lavas = number_of_lavas

        # Grid Value Encodings
        self.LAVA = -100
        self.EMPTY = 0
        self.BOX = 10
        self.BARRIER = 100

        # --- Action Space ---
        # Tuple(Target Coordinate, Action Type)
        # 1. Target Coordinate: [Row, Col] (MultiDiscrete)
        # 2. Action Type: 0=NoOp, 1-4=Move, 5=Barrier, 6=Hellify (Discrete)
        rows_limit = 100
        cols_limit = 100
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([rows_limit, cols_limit]),
            spaces.Discrete(7)
        ))

        # --- Observation Space ---
        # A Dict containing the full state observable by the agent.
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-100, high=100, shape=(rows_limit, cols_limit), dtype=np.int32),
            # Maps to Cursor Pos
            "agent": spaces.Box(low=0, high=rows_limit, shape=(2,), dtype=np.int32),
            "stamina": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "previous_selected_position": spaces.Box(low=0, high=rows_limit, shape=(2,), dtype=np.int32),
            "previous_action": spaces.Discrete(7)
        })

        self.gui = None

        # --- Internal State ---
        self.grid = None
        self.n_rows = 0  # Total rows (inner + border)
        self.n_cols = 0  # Total cols (inner + border)
        self.cursor_pos = np.array([0, 0], dtype=np.int32)
        self.stamina = 0.0
        self.timestep = 0

        # Physics: Tracks previous movement direction per cell for inertia calculation
        self.box_momentum_mask = None
        # Logic: List of detected Perfect Squares: {'top_left':(r,c), 'n':size, 'age':int}
        self.perfect_squares = []

        # --- Statistics Tracking ---
        self.boxes_destroyed_total = 0

        # Step-specific stats (reset every step)
        self.last_step_valid = False
        self.last_step_k = 0
        self.last_step_initial_force = False
        self.last_step_lava_destroyed = 0

        # Apply seed immediately if provided
        if seed is not None:
            self.reset(seed=seed)

    def _get_obs(self):
        """Returns the current state dictionary."""
        return {
            "grid": self.grid.copy(),
            "agent": np.array(self.cursor_pos, dtype=np.int32),
            "stamina": np.array([self.stamina], dtype=np.float32),
            "previous_selected_position": self.cursor_pos,
            "previous_action": 0  # Placeholder, typically updated in wrapper or external logic
        }

    def _get_info(self):
        """
        Constructs the info dictionary containing detailed game statistics.
        Useful for debugging or reward shaping.
        """
        # Count current boxes on grid
        current_boxes = np.sum(
            self.grid == self.BOX) if self.grid is not None else 0

        return {
            "timestep": self.timestep,
            "max_timestep": self.max_timestep,
            "stamina": self.stamina,
            "number_of_boxes": int(current_boxes),
            "number_destroyed": self.boxes_destroyed_total,
            "last_action_valid": self.last_step_valid,
            "chain_length_k": self.last_step_k,
            "initial_force_charged": bool(self.last_step_initial_force),
            "lava_destroyed_this_step": self.last_step_lava_destroyed,
            # Provide (Size, Location) of available squares
            "perfect_squares_available": [(ps['n'], ps['top_left']) for ps in self.perfect_squares]
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.

        1. Seeds RNG.
        2. Resets Stamina and stats.
        3. Loads or generates a map.
        4. Resets physics (momentum) and logic (perfect squares).
        """
        super().reset(seed=seed)

        self.timestep = 0
        self.stamina = self.initial_stamina_cap
        self.boxes_destroyed_total = 0

        self._reset_step_stats()

        # Create the map
        self._load_or_generate_map()

        # Clear physics state
        self.box_momentum_mask = np.zeros_like(self.grid, dtype=np.int32)
        self.perfect_squares = []

        # Ensure cursor is within bounds (e.g. if map size changed)
        if not (0 <= self.cursor_pos[0] < self.n_rows and 0 <= self.cursor_pos[1] < self.n_cols):
            self.cursor_pos = np.array([1, 1], dtype=np.int32)

        if self.render_mode == "human":
            self._init_gui()

        return self._get_obs(), self._get_info()

    def _reset_step_stats(self):
        """Helper to reset transient statistics for the current step."""
        self.last_step_valid = False
        self.last_step_k = 0
        self.last_step_initial_force = False
        self.last_step_lava_destroyed = 0

    def _load_or_generate_map(self):
        """
        Loads a map from a text file or generates a random one.
        Then, pads the map with a 1-cell border of Lava.
        """
        raw_grid = []
        agent_start_raw = None

        # 1. Attempt to load from file
        if self.map_path:
            target_path = self.map_path
            # Path resolution logic
            if not os.path.exists(target_path):
                if os.path.exists(os.path.join(os.getcwd(), target_path)):
                    target_path = os.path.join(os.getcwd(), target_path)
                else:
                    target_path = None
                    print(
                        f"Warning: Map '{self.map_path}' not found. Switching to generator.")

            if target_path:
                print(f"Loading map from: {target_path}")
                try:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for r, line in enumerate(lines):
                        row_data = []
                        tokens = line.strip().split()
                        if not tokens:
                            continue
                        for c, token in enumerate(tokens):
                            # Parse tokens
                            val = self.EMPTY
                            if token == 'A':
                                agent_start_raw = (r, c)
                                val = self.EMPTY
                            elif token == '.':
                                val = self.EMPTY
                            elif token == 'B':
                                val = self.BOX
                            elif token == '#':
                                val = self.BARRIER
                            elif token == 'L':
                                val = self.LAVA
                            else:
                                try:
                                    val = int(token)
                                except:
                                    val = self.EMPTY
                            row_data.append(val)
                        raw_grid.append(row_data)

                    # Validate that the map is rectangular (not jagged)
                    if raw_grid:
                        width = len(raw_grid[0])
                        for row in raw_grid:
                            if len(row) != width:
                                raise ValueError(
                                    "Map rows have inconsistent lengths")

                except Exception as e:
                    print(f"Error parsing map: {e}. Switching to generator.")
                    raw_grid = []

        # 2. Fallback to Random Generator
        if not raw_grid:
            raw_grid, agent_start_raw = self._generate_random_grid_data()

        # 3. Apply Lava Border
        # Convert inner map to numpy
        inner_map = np.array(raw_grid, dtype=np.int32)
        h, w = inner_map.shape

        # Create larger grid filled with Lava
        self.n_rows = h + 2
        self.n_cols = w + 2
        self.grid = np.full((self.n_rows, self.n_cols),
                            self.LAVA, dtype=np.int32)

        # Embed inner map
        self.grid[1:1+h, 1:1+w] = inner_map

        # Set Cursor Position (Offset by 1 due to border)
        if agent_start_raw:
            self.cursor_pos = np.array(
                [agent_start_raw[0]+1, agent_start_raw[1]+1], dtype=np.int32)
        else:
            # Default to first empty spot or (1,1)
            empties = np.argwhere(self.grid == self.EMPTY)
            if len(empties) > 0:
                self.cursor_pos = empties[0].astype(np.int32)
            else:
                self.cursor_pos = np.array([1, 1], dtype=np.int32)

    def _generate_random_grid_data(self):
        """
        Generates a random grid layout based on initialization parameters.
        Uses self.np_random (seeded by Gym) for reproducibility.
        """
        rows = self.inner_rows
        cols = self.inner_cols
        grid = np.zeros((rows, cols), dtype=np.int32)

        # Generate all coordinates and shuffle them to place items randomly
        coords = [(r, c) for r in range(rows) for c in range(cols)]
        self.np_random.shuffle(coords)

        idx = 0
        total_cells = rows * cols

        # Place Agent (Cursor)
        agent_start = (0, 0)
        if idx < total_cells:
            agent_start = coords[idx]
            idx += 1

        # Place Boxes
        count = 0
        while count < self.n_boxes and idx < total_cells:
            r, c = coords[idx]
            grid[r, c] = self.BOX
            idx += 1
            count += 1

        # Place Barriers
        count = 0
        while count < self.n_barriers and idx < total_cells:
            r, c = coords[idx]
            grid[r, c] = self.BARRIER
            idx += 1
            count += 1

        # Place Internal Lava
        count = 0
        while count < self.n_lavas and idx < total_cells:
            r, c = coords[idx]
            grid[r, c] = self.LAVA
            idx += 1
            count += 1

        return grid, agent_start

    def step(self, action):
        """
        Advances the environment by one step.

        Flow:
        1. Update Cursor.
        2. Update Perfect Squares (Aging/Dissolution).
        3. Perform Action (Push, Barrier, or Hellify).
        4. Update Momentum.
        5. Check Termination.
        """
        self._reset_step_stats()

        target_pos, act_type = action
        self.cursor_pos = np.array(target_pos, dtype=np.int32)

        self.timestep += 1
        reward = 0
        terminated = False
        truncated = False

        # --- 1. Perfect Squares Logic ---

        # Detect squares in current configuration
        current_sq_configs = self._detect_perfect_squares()
        new_tracked = []

        # Match with previously tracked squares to increment Age
        for cfg in current_sq_configs:
            match = None
            for old in self.perfect_squares:
                if old['top_left'] == cfg['top_left'] and old['n'] == cfg['n']:
                    match = old
                    break
            if match:
                match['age'] += 1
                new_tracked.append(match)
            else:
                new_tracked.append(
                    {'top_left': cfg['top_left'], 'n': cfg['n'], 'age': 0})
        self.perfect_squares = new_tracked

        # Check for Age Limit (Dissolution)
        remaining_squares = []
        squares_changed = False
        for ps in self.perfect_squares:
            if ps['age'] >= self.perf_sq_limit:
                # Dissolve: Turn into Empty space
                r, c = ps['top_left']
                n = ps['n']
                self.grid[r:r+n, c:c+n] = self.EMPTY
                squares_changed = True
                self.boxes_destroyed_total += (n * n)
            else:
                remaining_squares.append(ps)
        self.perfect_squares = remaining_squares

        # If squares dissolved, re-scan to update state
        if squares_changed:
            current_sq_configs = self._detect_perfect_squares()
            self.perfect_squares = [
                {'top_left': x['top_left'], 'n': x['n'], 'age': 0} for x in current_sq_configs]

        # --- 2. Action Execution ---

        next_momentum_mask = np.zeros_like(self.box_momentum_mask)
        deltas = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}

        # Case A: Movement / Push (1-4)
        if 1 <= act_type <= 4:
            dr, dc = deltas[act_type]
            r, c = self.cursor_pos
            # Bounds check
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                # Only push if cursor is ON a box
                if self.grid[r, c] == self.BOX:
                    success = self._handle_push(
                        r, c, dr, dc, act_type, next_momentum_mask)
                    if success:
                        self.last_step_valid = True
                        # Cursor follows the box
                        new_r, new_c = r + dr, c + dc
                        if 0 <= new_r < self.n_rows and 0 <= new_c < self.n_cols:
                            if self.grid[new_r, new_c] == self.BOX:
                                self.cursor_pos = np.array(
                                    [new_r, new_c], dtype=np.int32)

        # Case B: Barrier Maker (5)
        elif act_type == 5:
            # Target oldest square n >= 2
            candidates = [ps for ps in self.perfect_squares if ps['n'] >= 2]
            candidates.sort(key=lambda x: x['age'], reverse=True)
            if candidates:
                self.last_step_valid = True
                t = candidates[0]
                r, c, n = t['top_left'][0], t['top_left'][1], t['n']
                self.grid[r:r+n, c:c+n] = self.BARRIER
                self.stamina += self.unit_force * (n * n)  # Reward: +10*n^2 by default (spec sec. 14)
                self.perfect_squares = []  # Clear list as grid changed

        # Case C: Hellify (6)
        elif act_type == 6:
            # Target oldest square n > 2
            candidates = [ps for ps in self.perfect_squares if ps['n'] > 2]
            candidates.sort(key=lambda x: x['age'], reverse=True)
            if candidates:
                self.last_step_valid = True
                t = candidates[0]
                r, c, n = t['top_left'][0], t['top_left'][1], t['n']
                # Outer border -> Empty, Inner -> Lava
                self.grid[r:r+n, c:c+n] = self.EMPTY
                if n > 2:
                    self.grid[r+1:r+n-1, c+1:c+n-1] = self.LAVA

                self.boxes_destroyed_total += (n * n)
                self.perfect_squares = []

        self.box_momentum_mask = next_momentum_mask

        # --- 3. Termination ---
        if self.stamina <= 0:
            terminated = True
        if self.timestep >= self.max_timestep:
            truncated = True
        if np.sum(self.grid == self.BOX) == 0:
            terminated = True

        obs = self._get_obs()
        obs["previous_action"] = act_type
        return obs, reward, terminated, truncated, self._get_info()

    def _handle_push(self, r, c, dr, dc, direction_code, next_momentum_mask):
        """
        Handles the physics of pushing a chain of boxes.

        Logic:
        1. Determine chain length (k).
        2. Check if the cell AFTER the chain is free (Empty/Lava).
        3. Apply Stamina Cost (Static vs Kinetic friction).
        4. Move boxes (handling Lava destruction).
        """
        # 1. Identify Chain
        chain_len = 0
        cr, cc = r, c
        while 0 <= cr < self.n_rows and 0 <= cc < self.n_cols and self.grid[cr, cc] == self.BOX:
            chain_len += 1
            cr += dr
            cc += dc

        tr, tc = cr, cc  # Target cell after chain
        can_push = False
        if 0 <= tr < self.n_rows and 0 <= tc < self.n_cols:
            if self.grid[tr, tc] == self.EMPTY or self.grid[tr, tc] == self.LAVA:
                can_push = True

        self.last_step_k = chain_len

        if can_push:
            # 2. Calculate Cost
            # Check previous momentum for the HEAD box
            is_stationary = (self.box_momentum_mask[r, c] != direction_code)
            self.last_step_initial_force = is_stationary

            cost = (self.unit_force * chain_len) + \
                (self.initial_force if is_stationary else 0)
            self.stamina -= cost

            # 3. Move Boxes
            # Handle head falling into Lava
            if self.grid[tr, tc] == self.LAVA:
                self.stamina += self.initial_force  # Refund
                self.last_step_lava_destroyed += 1
                self.boxes_destroyed_total += 1
            else:
                self.grid[tr, tc] = self.BOX
                next_momentum_mask[tr, tc] = direction_code

            # Move the rest of the chain
            for k in range(1, chain_len):
                dest_r = r + (chain_len - k) * dr
                dest_c = c + (chain_len - k) * dc
                self.grid[dest_r, dest_c] = self.BOX
                next_momentum_mask[dest_r, dest_c] = direction_code

            # Original cell becomes empty
            self.grid[r, c] = self.EMPTY
            return True

        return False

    def _detect_perfect_squares(self):
        """
        Scans grid for NxN blocks of boxes that are isolated from other boxes.
        Returns a list of dicts describing the squares.
        """
        squares = []
        box_mask = (self.grid == self.BOX).astype(np.int32)
        rows, cols = box_mask.shape
        visited = np.zeros_like(box_mask, dtype=bool)

        for r in range(rows - 1):
            for c in range(cols - 1):
                if box_mask[r, c] == 1 and not visited[r, c]:
                    max_n = 0
                    # Find max square size from top-left (r,c)
                    for k in range(2, min(rows-r, cols-c) + 1):
                        block = box_mask[r:r+k, c:c+k]
                        if np.all(block == 1):
                            max_n = k
                        else:
                            break

                    if max_n >= 2:
                        # Check isolation
                        if self._is_isolated(r, c, max_n, box_mask):
                            squares.append({'top_left': (r, c), 'n': max_n})
                            visited[r:r+max_n, c:c+max_n] = True
        return squares

    def _is_isolated(self, r, c, n, mask):
        """
        Checks if an NxN patch of boxes is surrounded by non-boxes.
        """
        rows, cols = mask.shape
        r_start, r_end = max(0, r - 1), min(rows, r + n + 1)
        c_start, c_end = max(0, c - 1), min(cols, c + n + 1)
        patch = mask[r_start:r_end, c_start:c_end]
        # The sum must equal exactly the boxes in the square (n*n), implying the border is 0.
        return np.sum(patch) == (n * n)

    def render(self):
        """
        Renders the environment.
        - 'human': Displays Pygame window.
        - 'rgb_array': Returns numpy array of pixels.
        """
        if self.render_mode == "human":
            if self.gui is None:
                self._init_gui()
            return self.gui.draw(self)
        elif self.render_mode == "rgb_array":
            if self.gui is None:
                self._init_gui()
            # Draw without flipping to screen
            self.gui.draw(self, flip=False)
            rgb_array = pygame.surfarray.array3d(self.gui.screen)
            # Transpose to (Height, Width, Channels) if needed, or standard Gym (H,W,C)
            return np.transpose(rgb_array, (1, 0, 2))

    def _init_gui(self):
        """Lazy initialization of the GUI module."""
        from gui import ShoverGUI
        self.gui = ShoverGUI(self.n_rows, self.n_cols)

    def close(self):
        """Clean up resources."""
        if self.gui:
            self.gui.close()
