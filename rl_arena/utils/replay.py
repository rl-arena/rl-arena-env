"""Replay recording and playback functionality."""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from rl_arena.core.types import ActionType, ObservationType, RewardType


class ReplayRecorder:
    """
    Records environment interactions for later playback and analysis.
    
    Replays are saved in JSON format with the following structure:
    {
        "metadata": {
            "environment": "pong",
            "timestamp": "2024-01-01T12:00:00",
            "num_players": 2,
            "configuration": {...}
        },
        "steps": [
            {
                "step": 0,
                "actions": [0, 1],
                "observations": [...],
                "rewards": [0.0, 0.0],
                "terminated": false,
                "truncated": false,
                "info": {...}
            },
            ...
        ]
    }
    """
    
    def __init__(
        self,
        environment_name: str,
        configuration: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the replay recorder.
        
        Args:
            environment_name: Name of the environment being recorded
            configuration: Environment configuration used
            metadata: Additional metadata to store with the replay
        """
        self.environment_name = environment_name
        self.configuration = configuration or {}
        self.metadata = metadata or {}
        self.steps: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def record_step(
        self,
        step: int,
        actions: List[ActionType],
        observations: List[ObservationType],
        rewards: List[RewardType],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """
        Record a single step of the environment.
        
        Args:
            step: Step number
            actions: Actions taken by all players
            observations: Observations received by all players
            rewards: Rewards received by all players
            terminated: Whether the episode terminated naturally
            truncated: Whether the episode was truncated
            info: Additional information from the environment
        """
        step_data = {
            "step": step,
            "actions": self._serialize(actions),
            "observations": self._serialize(observations),
            "rewards": self._serialize(rewards),
            "terminated": terminated,
            "truncated": truncated,
            "info": self._serialize(info),
        }
        self.steps.append(step_data)
    
    def save(self, filepath: str) -> None:
        """
        Save the replay to a JSON file.
        
        Args:
            filepath: Path where to save the replay
        """
        replay_data = {
            "metadata": {
                "environment": self.environment_name,
                "timestamp": self.start_time.isoformat(),
                "num_steps": len(self.steps),
                "configuration": self.configuration,
                **self.metadata,
            },
            "steps": self.steps,
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(replay_data, f, indent=2)
    
    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: ReplayRecorder._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ReplayRecorder._serialize(item) for item in obj]
        else:
            return obj


def load_replay(filepath: str) -> Dict[str, Any]:
    """
    Load a replay from a JSON file.
    
    Args:
        filepath: Path to the replay file
        
    Returns:
        Dictionary containing replay data with 'metadata' and 'steps' keys
        
    Example:
        >>> replay = load_replay("replays/game_001.json")
        >>> print(f"Environment: {replay['metadata']['environment']}")
        >>> print(f"Number of steps: {replay['metadata']['num_steps']}")
        >>> for step_data in replay['steps']:
        ...     print(f"Step {step_data['step']}: actions={step_data['actions']}")
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_replay(
    replay_data: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Save replay data to a JSON file.
    
    Args:
        replay_data: Replay data dictionary
        filepath: Path where to save the replay
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(replay_data, f, indent=2)


def replay_episode(
    environment: Any,
    replay_data: Dict[str, Any],
    render: bool = True,
) -> None:
    """
    Replay a recorded episode in an environment.
    
    Args:
        environment: The environment to replay in
        replay_data: Replay data loaded from a file
        render: Whether to render during replay
        
    Example:
        >>> import rl_arena
        >>> from rl_arena.utils import load_replay, replay_episode
        >>> 
        >>> env = rl_arena.make("pong")
        >>> replay = load_replay("replays/game_001.json")
        >>> replay_episode(env, replay, render=True)
    """
    # TODO: Implement replay playback
    # This would step through the recorded actions and visualize the game
    pass
