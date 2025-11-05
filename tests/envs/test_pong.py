"""Tests for Pong environment."""

import pytest
import numpy as np
import rl_arena
from rl_arena.core.exceptions import InvalidActionError, GameEndedError


def test_pong_creation(pong_env):
    """Test Pong environment creation."""
    assert pong_env.num_players == 2
    assert pong_env.action_space.n == 3  # UP, STAY, DOWN


def test_pong_reset(pong_env, seed):
    """Test Pong reset functionality."""
    observations, info = pong_env.reset(seed=seed)
    
    assert len(observations) == 2
    assert "scores" in info
    assert info["scores"] == [0, 0]
    
    # Check observation shape
    for obs in observations:
        assert obs.shape == (8,)
        assert isinstance(obs, np.ndarray)


def test_pong_step(pong_env, seed, sample_actions):
    """Test Pong step functionality."""
    pong_env.reset(seed=seed)
    
    observations, rewards, terminated, truncated, info = pong_env.step(sample_actions)
    
    assert len(observations) == 2
    assert len(rewards) == 2
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "scores" in info
    assert "ball_pos" in info


def test_pong_invalid_action(pong_env, seed):
    """Test that invalid actions raise errors."""
    pong_env.reset(seed=seed)
    
    # Invalid action value
    with pytest.raises(InvalidActionError):
        pong_env.step([5, 1])


def test_pong_wrong_number_of_actions(pong_env, seed):
    """Test that wrong number of actions raises an error."""
    pong_env.reset(seed=seed)
    
    with pytest.raises(ValueError, match="Expected 2 actions"):
        pong_env.step([1])


def test_pong_episode_completion(pong_env_configured, seed):
    """Test that episodes complete correctly."""
    pong_env_configured.reset(seed=seed)
    
    max_steps = 200
    for _ in range(max_steps):
        actions = [1, 1]  # Both stay
        observations, rewards, terminated, truncated, info = pong_env_configured.step(actions)
        
        if terminated or truncated:
            break
    
    assert terminated or truncated
    assert pong_env_configured.is_done


def test_pong_scoring(pong_env, seed):
    """Test that scoring works correctly."""
    observations, info = pong_env.reset(seed=seed)
    initial_scores = info["scores"].copy()
    
    # Play until someone scores
    scored = False
    for _ in range(1000):
        actions = [1, 1]  # Both stay
        observations, rewards, terminated, truncated, info = pong_env.step(actions)
        
        # Check if anyone scored
        if any(r != 0 for r in rewards):
            scored = True
            # Verify that scores increased
            assert sum(info["scores"]) > sum(initial_scores)
            # Verify rewards are correct
            if rewards[0] > 0:
                assert info["scores"][0] > initial_scores[0]
            if rewards[1] > 0:
                assert info["scores"][1] > initial_scores[1]
            break
        
        if terminated or truncated:
            break
    
    # In 1000 steps, someone should have scored
    assert scored


def test_pong_paddle_movement(pong_env, seed):
    """Test that paddle movement works."""
    pong_env.reset(seed=seed)
    
    # Get initial paddle positions
    obs_initial, _, _, _, _ = pong_env.step([1, 1])  # Both STAY
    initial_paddle1_y = obs_initial[0][4]
    initial_paddle2_y = obs_initial[0][5]
    
    # Move paddles
    for _ in range(5):
        obs, _, _, _, _ = pong_env.step([0, 2])  # P1 UP, P2 DOWN
    
    # Check that paddles moved
    final_paddle1_y = obs[0][4]
    final_paddle2_y = obs[0][5]
    
    assert final_paddle1_y < initial_paddle1_y  # Moved up
    assert final_paddle2_y > initial_paddle2_y  # Moved down


def test_pong_ball_movement(pong_env, seed):
    """Test that the ball moves."""
    observations, _ = pong_env.reset(seed=seed)
    initial_ball_pos = observations[0][:2].copy()
    
    # Step a few times
    for _ in range(10):
        observations, _, terminated, truncated, _ = pong_env.step([1, 1])
        if terminated or truncated:
            break
    
    final_ball_pos = observations[0][:2]
    
    # Ball should have moved
    assert not np.allclose(initial_ball_pos, final_ball_pos)


def test_pong_get_observation(pong_env, seed):
    """Test getting observations for specific players."""
    pong_env.reset(seed=seed)
    
    obs0 = pong_env.get_observation(0)
    obs1 = pong_env.get_observation(1)
    
    assert obs0 is not None
    assert obs1 is not None
    assert obs0.shape == (8,)
    assert obs1.shape == (8,)


def test_pong_render_ansi(pong_env, seed):
    """Test ANSI rendering."""
    pong_env.reset(seed=seed)
    
    output = pong_env.render(mode="ansi")
    
    assert isinstance(output, str)
    assert len(output) > 0


def test_pong_configuration(pong_env_configured):
    """Test custom configuration."""
    assert pong_env_configured.winning_score == 3
    assert pong_env_configured.max_steps == 100


def test_pong_step_after_done(pong_env, seed):
    """Test that stepping after episode ends raises an error."""
    # Create a short episode
    env = rl_arena.make("pong", configuration={"winning_score": 1, "max_steps": 50})
    env.reset(seed=seed)
    
    # Play until done
    for _ in range(1000):
        observations, rewards, terminated, truncated, info = env.step([1, 1])
        if terminated or truncated:
            break
    
    # Try to step after done
    with pytest.raises(GameEndedError):
        env.step([1, 1])
    
    env.close()


def test_pong_determinism(seed):
    """Test that episodes are deterministic with the same seed."""
    def run_episode(seed_val):
        env = rl_arena.make("pong")
        observations, _ = env.reset(seed=seed_val)
        
        trajectory = [observations[0].copy()]
        for _ in range(50):
            observations, _, terminated, truncated, _ = env.step([1, 1])
            trajectory.append(observations[0].copy())
            if terminated or truncated:
                break
        
        env.close()
        return trajectory
    
    # Run two episodes with the same seed
    traj1 = run_episode(seed)
    traj2 = run_episode(seed)
    
    # They should be identical
    assert len(traj1) == len(traj2)
    for obs1, obs2 in zip(traj1, traj2):
        assert np.allclose(obs1, obs2)


def test_environment_seed_method(pong_env):
    """Test the seed() method directly."""
    # Test seeding returns the seed value
    result = pong_env.seed(42)
    assert result == [42]
    
    # Test that np_random is initialized
    assert pong_env.np_random is not None
    
    # Test reproducibility
    pong_env.seed(100)
    random1 = pong_env.np_random.random()
    
    pong_env.seed(100)
    random2 = pong_env.np_random.random()
    
    assert random1 == random2
    
    # Test with None
    result = pong_env.seed(None)
    assert result == []


def test_environment_np_random_property(pong_env):
    """Test the np_random property lazy initialization."""
    # Before any seeding, np_random should still work (non-deterministic)
    env = rl_arena.make("pong")
    
    # Access np_random before seeding - should initialize automatically
    rng = env.np_random
    assert rng is not None
    assert isinstance(rng, np.random.Generator)
    
    # Should be able to generate random numbers
    value = rng.random()
    assert 0.0 <= value <= 1.0
    
    env.close()

