"""Rendering functionality for the Pong environment."""

from typing import Optional, Any
import numpy as np


class PongRenderer:
    """
    Renderer for the Pong environment.

    Supports multiple rendering modes:
    - 'human': Display in a window (requires matplotlib or pygame)
    - 'rgb_array': Return an RGB numpy array
    - 'ansi': Return an ASCII string representation
    """

    def __init__(self, width: float = 1.0, height: float = 1.0, mode: str = "human"):
        """
        Initialize the renderer.

        Args:
            width: Field width
            height: Field height
            mode: Rendering mode ('human', 'rgb_array', or 'ansi')
        """
        self.width = width
        self.height = height
        self.mode = mode
        self.fig = None
        self.ax = None

        if mode == "human":
            try:
                import matplotlib.pyplot as plt

                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(10, 6))
                self.plt = plt
            except ImportError:
                print("Warning: matplotlib not installed. Falling back to ANSI rendering.")
                self.mode = "ansi"

    def render(
        self,
        ball_pos: np.ndarray,
        ball_radius: float,
        paddle1_y: float,
        paddle2_y: float,
        paddle_height: float,
        score1: int,
        score2: int,
    ) -> Optional[Any]:
        """
        Render the current game state.

        Args:
            ball_pos: Ball position [x, y]
            ball_radius: Ball radius
            paddle1_y: Player 1 paddle Y position
            paddle2_y: Player 2 paddle Y position
            paddle_height: Paddle height
            score1: Player 1 score
            score2: Player 2 score

        Returns:
            - None for 'human' mode
            - numpy array for 'rgb_array' mode
            - string for 'ansi' mode
        """
        if self.mode == "ansi":
            return self._render_ansi(ball_pos, paddle1_y, paddle2_y, paddle_height, score1, score2)
        elif self.mode == "human":
            return self._render_human(
                ball_pos, ball_radius, paddle1_y, paddle2_y, paddle_height, score1, score2
            )
        elif self.mode == "rgb_array":
            return self._render_rgb_array(
                ball_pos, ball_radius, paddle1_y, paddle2_y, paddle_height, score1, score2
            )

    def _render_ansi(
        self,
        ball_pos: np.ndarray,
        paddle1_y: float,
        paddle2_y: float,
        paddle_height: float,
        score1: int,
        score2: int,
    ) -> str:
        """Render as ASCII art."""
        # Create a character grid
        grid_width = 60
        grid_height = 20
        grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

        # Draw borders
        for i in range(grid_height):
            grid[i][0] = "|"
            grid[i][grid_width - 1] = "|"
        for i in range(grid_width):
            grid[0][i] = "-"
            grid[grid_height - 1][i] = "-"

        # Draw ball
        ball_x = int(ball_pos[0] * (grid_width - 2)) + 1
        ball_y = int(ball_pos[1] * (grid_height - 2)) + 1
        if 0 < ball_y < grid_height and 0 < ball_x < grid_width:
            grid[ball_y][ball_x] = "O"

        # Draw paddles
        paddle_h_cells = max(1, int(paddle_height * grid_height))

        # Player 1 paddle (left)
        paddle1_center = int(paddle1_y * (grid_height - 2)) + 1
        for dy in range(-paddle_h_cells // 2, paddle_h_cells // 2 + 1):
            y = paddle1_center + dy
            if 0 < y < grid_height:
                grid[y][2] = "█"

        # Player 2 paddle (right)
        paddle2_center = int(paddle2_y * (grid_height - 2)) + 1
        for dy in range(-paddle_h_cells // 2, paddle_h_cells // 2 + 1):
            y = paddle2_center + dy
            if 0 < y < grid_height:
                grid[y][grid_width - 3] = "█"

        # Convert grid to string
        result = f"\n  Player 1: {score1}  |  Player 2: {score2}\n"
        result += "\n".join(["".join(row) for row in grid])
        result += "\n"

        return result

    def _render_human(
        self,
        ball_pos: np.ndarray,
        ball_radius: float,
        paddle1_y: float,
        paddle2_y: float,
        paddle_height: float,
        score1: int,
        score2: int,
    ) -> None:
        """Render using matplotlib."""
        if self.ax is None:
            return

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"Player 1: {score1}  |  Player 2: {score2}", fontsize=16)

        # Draw field
        self.ax.axhline(y=0, color="white", linewidth=2)
        self.ax.axhline(y=self.height, color="white", linewidth=2)
        self.ax.axvline(x=self.width / 2, color="gray", linestyle="--", linewidth=1)

        # Draw ball
        circle = self.plt.Circle((ball_pos[0], ball_pos[1]), ball_radius, color="white", fill=True)
        self.ax.add_patch(circle)

        # Draw paddles
        paddle_width = 0.02

        # Player 1 paddle (left)
        paddle1 = self.plt.Rectangle(
            (0.05 - paddle_width / 2, paddle1_y - paddle_height / 2),
            paddle_width,
            paddle_height,
            color="blue",
            fill=True,
        )
        self.ax.add_patch(paddle1)

        # Player 2 paddle (right)
        paddle2 = self.plt.Rectangle(
            (0.95 - paddle_width / 2, paddle2_y - paddle_height / 2),
            paddle_width,
            paddle_height,
            color="red",
            fill=True,
        )
        self.ax.add_patch(paddle2)

        self.ax.set_facecolor("black")
        self.plt.pause(0.001)
        self.plt.draw()

    def _render_rgb_array(
        self,
        ball_pos: np.ndarray,
        ball_radius: float,
        paddle1_y: float,
        paddle2_y: float,
        paddle_height: float,
        score1: int,
        score2: int,
    ) -> np.ndarray:
        """Render as RGB array."""
        # TODO: Implement RGB array rendering
        # This would create a numpy array of shape (height, width, 3) with RGB values
        width, height = 640, 480
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # For now, return a placeholder
        # A full implementation would draw the game state onto this array
        return img

    def close(self) -> None:
        """Close the renderer and clean up resources."""
        if self.fig is not None:
            try:
                self.plt.close(self.fig)
            except:
                pass
            self.fig = None
            self.ax = None
