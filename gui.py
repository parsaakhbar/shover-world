import pygame
import numpy as np


class ShoverGUI:
    """
    Handles the graphical rendering and user input for the Shover World environment.

    This class uses Pygame to create a window, draw the grid state, visualize 
    the agent's cursor, and map keyboard/mouse inputs to environment actions.

    Attributes:
        n_rows (int): Number of grid rows.
        n_cols (int): Number of grid columns.
        cell_size (int): The pixel width/height of a single grid cell.
        width (int): Total window width.
        height (int): Total window height (includes UI header).
        screen (pygame.Surface): The main display surface.
        font (pygame.font.Font): Font used for the HUD info text.
    """

    def __init__(self, n_rows, n_cols, cell_size=40):
        """
        Initialize the GUI window.

        Args:
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            cell_size (int): Pixel size for each cell (default: 40).
        """
        pygame.init()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cell_size = cell_size

        # Calculate window dimensions
        self.width = n_cols * cell_size
        # Add 60 pixels of vertical space for the Info Bar (HUD) at the top
        self.height = n_rows * cell_size + 60

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Shover World")

        # Initialize font (System Arial, size 16)
        self.font = pygame.font.SysFont("Arial", 16)

        # --- Color Palette (R, G, B) ---
        self.COLOR_BG = (30, 30, 30)           # Dark Gray (Background)
        self.COLOR_GRID = (50, 50, 50)         # Lighter Gray (Grid Lines)
        self.COLOR_EMPTY = (200, 200, 200)     # Light Gray (Empty Cells)
        self.COLOR_BOX = (139, 69, 19)         # Saddle Brown (Boxes)
        self.COLOR_BARRIER = (50, 50, 50)      # Dark Slate (Barriers)
        self.COLOR_LAVA = (255, 69, 0)         # Red-Orange (Lava)
        self.COLOR_CURSOR_BOX = (255, 0, 0)    # Red (Cursor when on Box)
        # Yellow (Cursor when on Empty)
        self.COLOR_CURSOR_EMPTY = (255, 255, 0)

    def handle_input(self):
        """
        Processes Pygame events (Mouse clicks, Key presses) and converts them
        into action data.

        Returns:
            tuple: (action_type, target_pos)
                - action_type (int): 
                    0 (NoOp), 1-4 (Move), 5 (Barrier), 6 (Hellify), 
                    99 (Reset), -1 (Quit).
                - target_pos (tuple or None): (row, col) if mouse clicked, else None.
        """
        action_type = 0
        target_pos = None

        events = pygame.event.get()
        for event in events:
            # Handle Window Close button
            if event.type == pygame.QUIT:
                return -1, None

            # Handle Mouse Clicks (Selection)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                # Check if click is below the Info Bar (60px offset)
                if my > 60:
                    c = mx // self.cell_size
                    r = (my - 60) // self.cell_size  # Adjust for header

                    # Verify click is within grid bounds
                    if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                        target_pos = (r, c)

            # Handle Keyboard Inputs (Actions)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    action_type = 1  # Push Up
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    action_type = 2  # Push Right
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    action_type = 3  # Push Down
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    action_type = 4  # Push Left
                elif event.key == pygame.K_b:
                    action_type = 5  # Special: Barrier Maker
                elif event.key == pygame.K_h:
                    action_type = 6  # Special: Hellify
                elif event.key == pygame.K_r:
                    action_type = 99  # Reset Environment
                elif event.key == pygame.K_q:
                    return -1, None  # Quit App

        return action_type, target_pos

    def draw(self, env, flip=True):
        """
        Renders the current state of the environment to the screen.

        Args:
            env (ShoverWorldEnv): The environment instance containing the
                                  grid, stamina, and other state data.
            flip (bool): Update the display after drawing. Pass False when
                         only the off-screen surface is needed (rgb_array).
        """
        self.screen.fill(self.COLOR_BG)

        # --- Draw Info Bar (HUD) ---
        # Line 1: Timestep, Stamina, Box Count
        info_text = f"Step: {env.timestep}/{env.max_timestep} | Stamina: {env.stamina:.0f} | Boxes: {np.sum(env.grid==10)}"
        text_surf = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        # Line 2: Perfect Square Status
        sq_text = f"Perf Squares: {len(env.perfect_squares)}"
        sq_surf = self.font.render(sq_text, True, (200, 200, 255))
        self.screen.blit(sq_surf, (10, 35))

        # --- Draw Grid ---
        y_offset = 60  # Start drawing below the HUD

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # Define the rectangle for this cell
                rect = pygame.Rect(c * self.cell_size,
                                   y_offset + r * self.cell_size,
                                   self.cell_size,
                                   self.cell_size)

                val = env.grid[r, c]

                # Determine color based on cell content
                color = self.COLOR_EMPTY
                if val == 10:
                    color = self.COLOR_BOX
                elif val == 100:
                    color = self.COLOR_BARRIER
                elif val == -100:
                    color = self.COLOR_LAVA

                # Draw cell fill
                pygame.draw.rect(self.screen, color, rect)
                # Draw cell border
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # --- Draw Cursor Highlight ---
                if env.cursor_pos[0] == r and env.cursor_pos[1] == c:
                    # Red outline if on a Box (indicating "Ready to Push")
                    # Yellow outline if on Empty/Lava (Navigation only)
                    highlight = self.COLOR_CURSOR_BOX if val == 10 else self.COLOR_CURSOR_EMPTY
                    pygame.draw.rect(self.screen, highlight, rect, 3)

        # Update the full display surface to the screen
        if flip:
            pygame.display.flip()

    def close(self):
        """
        Cleanly shuts down the Pygame instance.
        """
        pygame.quit()
