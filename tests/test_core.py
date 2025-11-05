"""Tests for core Environment base class."""

import pytest
import numpy as np
from abc import ABC
from rl_arena.core.environment import Environment
from rl_arena.core.exceptions import InvalidPlayerError
import gymnasium as gym


class DummyEnvironment(Environment):
    """Minimal environment implementation for testing."""

    @property
    def num_players(self):
        return 2

    @property
    def action_space(self):
        return gym.spaces.Discrete(3)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._current_step = 0
        self._done = False
        obs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        return [obs, obs], {}

    def step(self, actions):
        self._current_step += 1
        obs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        rewards = [0.0, 0.0]
        terminated = False
        truncated = self._current_step >= 10
        self._done = terminated or truncated
        return [obs, obs], rewards, terminated, truncated, {}

    def render(self, mode="human"):
        return "dummy render"

    def get_observation(self, player_id):
        if player_id not in [0, 1]:
            raise InvalidPlayerError(player_id, self.num_players)
        return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)


def test_environment_is_abstract():
    """Test that Environment cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Environment()


def test_dummy_environment_creation():
    """Test that a concrete implementation can be created."""
    env = DummyEnvironment()
    assert env.num_players == 2
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)


def test_environment_reset():
    """Test environment reset functionality."""
    env = DummyEnvironment()
    observations, info = env.reset(seed=42)

    assert len(observations) == 2
    assert env.current_step == 0
    assert not env.is_done


def test_environment_step():
    """Test environment step functionality."""
    env = DummyEnvironment()
    env.reset()

    observations, rewards, terminated, truncated, info = env.step([0, 1])

    assert len(observations) == 2
    assert len(rewards) == 2
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.current_step == 1


def test_environment_episode_length():
    """Test that environment respects episode length limits."""
    env = DummyEnvironment()
    env.reset()

    for i in range(15):
        observations, rewards, terminated, truncated, info = env.step([0, 1])
        if terminated or truncated:
            break

    assert env.current_step == 10
    assert truncated


def test_get_observation_valid_player():
    """Test getting observation for valid player IDs."""
    env = DummyEnvironment()
    env.reset()

    obs0 = env.get_observation(0)
    obs1 = env.get_observation(1)

    assert obs0 is not None
    assert obs1 is not None


def test_get_observation_invalid_player():
    """Test that invalid player IDs raise an error."""
    env = DummyEnvironment()
    env.reset()

    with pytest.raises(InvalidPlayerError):
        env.get_observation(2)

    with pytest.raises(InvalidPlayerError):
        env.get_observation(-1)


def test_environment_config():
    """Test environment configuration."""
    config = {"test_param": 42, "another_param": "value"}
    env = DummyEnvironment(configuration=config)

    retrieved_config = env.get_config()
    assert retrieved_config == config
    assert retrieved_config is not config  # Should be a copy


def test_environment_render():
    """Test environment rendering."""
    env = DummyEnvironment()
    env.reset()

    result = env.render(mode="human")
    assert result is not None


def test_environment_close():
    """Test environment cleanup."""
    env = DummyEnvironment()
    env.close()  # Should not raise an error
