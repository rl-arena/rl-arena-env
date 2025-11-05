"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import rl_arena
from rl_arena.core.agent import RandomAgent


@pytest.fixture
def pong_env():
    """Create a Pong environment for testing."""
    env = rl_arena.make("pong")
    yield env
    env.close()


@pytest.fixture
def pong_env_configured():
    """Create a configured Pong environment for testing."""
    config = {
        "winning_score": 3,
        "max_steps": 100,
        "ball_speed": 0.03,
    }
    env = rl_arena.make("pong", configuration=config)
    yield env
    env.close()


@pytest.fixture
def random_agents(pong_env):
    """Create two random agents for testing."""
    return [
        RandomAgent(pong_env.action_space, player_id=0),
        RandomAgent(pong_env.action_space, player_id=1),
    ]


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def sample_observation():
    """Create a sample observation for testing."""
    return np.array([0.5, 0.5, 0.01, 0.01, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def sample_actions():
    """Create sample actions for testing."""
    return [1, 1]  # Both players STAY
