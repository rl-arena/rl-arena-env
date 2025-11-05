"""Base Environment class for RL Arena."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from rl_arena.core.types import ObservationType, ActionType, RewardType


class Environment(ABC):
    """
    Abstract base class for all RL Arena environments.

    This class defines the interface that all environments must implement.
    It follows the Gymnasium (OpenAI Gym) API pattern for compatibility.

    Attributes:
        num_players: Number of players/agents in the environment
        action_space: The action space for each player
        observation_space: The observation space for each player
        metadata: Environment metadata (render modes, etc.)
    """

    metadata: Dict[str, Any] = {"render_modes": ["human", "rgb_array", "ansi"]}

    def __init__(self, configuration: Optional[Dict[str, Any]] = None):
        """
        Initialize the environment.

        Args:
            configuration: Optional dictionary of configuration parameters
        """
        self.configuration = configuration or {}
        self._num_players: int = 2  # Default to 2-player games
        self._current_step: int = 0
        self._done: bool = False
        self._np_random: Optional[np.random.Generator] = None
        self._seed: Optional[int] = None

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players in the environment."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """
        Return the action space for the environment.

        For multi-agent environments, this typically returns the action space
        for a single agent (assuming all agents have the same action space).
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        """
        Return the observation space for the environment.

        For multi-agent environments, this typically returns the observation
        space for a single agent.
        """
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[ObservationType], Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional reset parameters

        Returns:
            observations: List of initial observations for each player
            info: Dictionary with auxiliary information

        Example:
            >>> observations, info = env.reset(seed=42)
            >>> print(len(observations))  # Should equal num_players
            2
        """
        pass

    @abstractmethod
    def step(
        self, actions: List[ActionType]
    ) -> Tuple[List[ObservationType], List[RewardType], bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions: List of actions, one for each player

        Returns:
            observations: List of observations for each player after the step
            rewards: List of rewards for each player
            terminated: Whether the episode has ended (win/loss condition)
            truncated: Whether the episode was cut off (time limit, etc.)
            info: Dictionary with auxiliary information

        Example:
            >>> observations, rewards, terminated, truncated, info = env.step([action1, action2])
            >>> if terminated:
            ...     print(f"Game ended! Rewards: {rewards}")
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[Any]:
        """
        Render the environment.

        Args:
            mode: Rendering mode
                - 'human': Render to the current display or terminal
                - 'rgb_array': Return an RGB array (numpy array)
                - 'ansi': Return a string (ANSI) representation

        Returns:
            Depends on mode:
                - None for 'human' mode
                - numpy array for 'rgb_array' mode
                - string for 'ansi' mode
        """
        pass

    @abstractmethod
    def get_observation(self, player_id: int) -> ObservationType:
        """
        Get the current observation for a specific player.

        This method is useful for partially observable environments where
        each player sees a different view of the game state.

        Args:
            player_id: The ID of the player (0-indexed)

        Returns:
            The observation for the specified player

        Raises:
            ValueError: If player_id is invalid
        """
        pass

    def close(self) -> None:
        """
        Clean up environment resources.

        Override this method if your environment needs cleanup
        (e.g., closing windows, disconnecting from servers).
        """
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the random seed for the environment.

        Args:
            seed: The random seed

        Returns:
            List of seeds used by the environment's random number generators

        Example:
            >>> env.seed(42)
            [42]
            >>> # Now all random operations in the environment will be reproducible
        """
        self._seed = seed
        self._np_random = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    @property
    def np_random(self) -> np.random.Generator:
        """
        Get the numpy random number generator for this environment.

        This ensures lazy initialization of the RNG. If seed() hasn't been called,
        it initializes with None (non-deterministic).

        Returns:
            numpy random number generator

        Example:
            >>> # Use in subclasses for reproducible randomness
            >>> random_value = self.np_random.random()
            >>> random_int = self.np_random.integers(0, 10)
        """
        if self._np_random is None:
            self._np_random = np.random.default_rng(self._seed)
        return self._np_random

    @property
    def current_step(self) -> int:
        """Return the current step number."""
        return self._current_step

    @property
    def is_done(self) -> bool:
        """Return whether the episode is done."""
        return self._done

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the environment.

        Returns:
            Dictionary of configuration parameters
        """
        return self.configuration.copy()
