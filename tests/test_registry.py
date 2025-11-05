"""Tests for environment registry."""

import pytest
from rl_arena.envs.registry import (
    ENVIRONMENT_REGISTRY,
    register_environment,
    unregister_environment,
    get_environment_class,
)
from rl_arena.core.environment import Environment
from rl_arena.core.exceptions import InvalidPlayerError
import gymnasium as gym
import numpy as np


# Create a test environment
@register_environment("test-env")
class TestEnvironment(Environment):
    """Test environment for registry testing."""

    @property
    def num_players(self):
        return 2

    @property
    def action_space(self):
        return gym.spaces.Discrete(2)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = np.array([0.0, 0.0], dtype=np.float32)
        return [obs, obs], {}

    def step(self, actions):
        obs = np.array([0.0, 0.0], dtype=np.float32)
        return [obs, obs], [0.0, 0.0], False, False, {}

    def render(self, mode="human"):
        return None

    def get_observation(self, player_id):
        if player_id not in [0, 1]:
            raise InvalidPlayerError(player_id, self.num_players)
        return np.array([0.0, 0.0], dtype=np.float32)


def test_register_environment():
    """Test that environments can be registered."""
    assert "test-env" in ENVIRONMENT_REGISTRY
    assert ENVIRONMENT_REGISTRY["test-env"] == TestEnvironment


def test_register_duplicate_environment():
    """Test that registering duplicate environments raises an error."""
    with pytest.raises(ValueError, match="already registered"):

        @register_environment("test-env")
        class DuplicateEnvironment(Environment):
            pass


def test_register_non_environment_class():
    """Test that registering non-Environment classes raises an error."""
    with pytest.raises(TypeError, match="must be a subclass"):

        @register_environment("invalid-env")
        class NotAnEnvironment:
            pass


def test_get_environment_class():
    """Test getting environment class by name."""
    env_class = get_environment_class("test-env")
    assert env_class == TestEnvironment


def test_get_nonexistent_environment():
    """Test that getting a nonexistent environment raises an error."""
    with pytest.raises(KeyError, match="not found"):
        get_environment_class("nonexistent-env")


def test_unregister_environment():
    """Test unregistering an environment."""

    # Register a temporary environment
    @register_environment("temp-env")
    class TempEnvironment(TestEnvironment):
        pass

    assert "temp-env" in ENVIRONMENT_REGISTRY

    # Unregister it
    unregister_environment("temp-env")
    assert "temp-env" not in ENVIRONMENT_REGISTRY


def test_unregister_nonexistent_environment():
    """Test that unregistering a nonexistent environment raises an error."""
    with pytest.raises(KeyError, match="not registered"):
        unregister_environment("nonexistent-env")


def test_pong_registered():
    """Test that the Pong environment is registered."""
    assert "pong" in ENVIRONMENT_REGISTRY


def test_environment_instantiation_from_registry():
    """Test creating an environment instance from the registry."""
    env_class = ENVIRONMENT_REGISTRY["test-env"]
    env = env_class()

    assert isinstance(env, Environment)
    assert env.num_players == 2

    env.close()
