"""Tests for the make() factory function."""

import pytest
import rl_arena
from rl_arena.core.environment import Environment
from rl_arena.core.exceptions import EnvironmentNotFoundError


def test_make_pong():
    """Test creating a Pong environment with make()."""
    env = rl_arena.make("pong")
    
    assert isinstance(env, Environment)
    assert env.num_players == 2
    
    env.close()


def test_make_with_configuration():
    """Test creating an environment with custom configuration."""
    config = {
        "winning_score": 5,
        "max_steps": 500,
    }
    env = rl_arena.make("pong", configuration=config)
    
    assert env.configuration["winning_score"] == 5
    assert env.configuration["max_steps"] == 500
    
    env.close()


def test_make_nonexistent_environment():
    """Test that making a nonexistent environment raises an error."""
    with pytest.raises(EnvironmentNotFoundError, match="not found"):
        rl_arena.make("nonexistent-game")


def test_make_returns_correct_type():
    """Test that make() returns the correct environment type."""
    env = rl_arena.make("pong")
    
    # Should be an instance of Environment
    assert isinstance(env, Environment)
    
    # Should have all required methods
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "render")
    assert hasattr(env, "close")
    assert hasattr(env, "get_observation")
    
    # Should have required properties
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")
    assert hasattr(env, "num_players")
    
    env.close()


def test_make_empty_configuration():
    """Test that make() works with empty configuration."""
    env = rl_arena.make("pong", configuration={})
    
    assert isinstance(env, Environment)
    
    env.close()


def test_make_none_configuration():
    """Test that make() works with None configuration."""
    env = rl_arena.make("pong", configuration=None)
    
    assert isinstance(env, Environment)
    
    env.close()


def test_environment_functional_after_make():
    """Test that environment is functional after creation."""
    env = rl_arena.make("pong")
    
    # Test reset
    observations, info = env.reset(seed=42)
    assert len(observations) == 2
    
    # Test step
    actions = [env.action_space.sample(), env.action_space.sample()]
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    assert len(observations) == 2
    assert len(rewards) == 2
    
    env.close()
