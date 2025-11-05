"""Example: Training a DQN agent to play Pong.

This example demonstrates how to train a Deep Q-Network (DQN) agent
using stable-baselines3 to play Pong against a random opponent.

Requirements:
    pip install stable-baselines3[extra]
"""

import rl_arena
from rl_arena.core.agent import RandomAgent
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import gymnasium as gym
from typing import Any, Dict, List, Tuple


class PongWrapper(gym.Env):
    """
    Gymnasium wrapper for RL Arena Pong environment.
    
    This wrapper converts the multi-agent RL Arena environment into
    a single-agent Gymnasium environment for training with SB3.
    """
    
    def __init__(self, env_name: str = "pong", opponent_agent: Any = None):
        """
        Initialize the wrapper.
        
        Args:
            env_name: Name of the RL Arena environment
            opponent_agent: Agent to play against (default: RandomAgent)
        """
        super().__init__()
        
        self.env = rl_arena.make(env_name)
        self.opponent = opponent_agent or RandomAgent(
            self.env.action_space,
            player_id=1
        )
        
        # Single agent perspective
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        observations, info = self.env.reset(seed=seed, options=options)
        return observations[0], info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action for player 0 (our agent)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get opponent's action
        opponent_obs = self.env.get_observation(1)
        opponent_action = self.opponent.act(opponent_obs)
        
        # Step with both actions
        observations, rewards, terminated, truncated, info = self.env.step(
            [action, opponent_action]
        )
        
        # Return from player 0's perspective
        return observations[0], rewards[0], terminated, truncated, info
    
    def render(self, mode: str = "human") -> Any:
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def train_dqn_agent(
    total_timesteps: int = 100_000,
    save_path: str = "models/dqn_pong",
) -> DQN:
    """
    Train a DQN agent to play Pong.
    
    Args:
        total_timesteps: Total number of timesteps to train for
        save_path: Path to save the trained model
        
    Returns:
        Trained DQN model
    """
    print("Creating training environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: PongWrapper("pong")])
    env = VecMonitor(env)
    
    print("Initializing DQN agent...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/dqn_pong/",
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    
    return model


def evaluate_agent(
    model: DQN,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """
    Evaluate a trained agent.
    
    Args:
        model: Trained DQN model
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    
    env = PongWrapper("pong")
    
    wins = 0
    total_reward = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
        
        total_reward += episode_reward
        total_steps += episode_steps
        
        # Check if agent won
        if info['scores'][0] > info['scores'][1]:
            wins += 1
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, "
              f"Scores={info['scores']}, Steps={episode_steps}")
    
    env.close()
    
    metrics = {
        "win_rate": wins / num_episodes,
        "avg_reward": total_reward / num_episodes,
        "avg_steps": total_steps / num_episodes,
    }
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Average Steps: {metrics['avg_steps']:.1f}")
    
    return metrics


def main():
    """Main training and evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN agent for Pong")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new agent"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a trained agent"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of training timesteps (default: 100,000)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/dqn_pong",
        help="Path to save/load model"
    )
    
    args = parser.parse_args()
    
    if args.train:
        model = train_dqn_agent(
            total_timesteps=args.timesteps,
            save_path=args.model_path,
        )
        
        if args.eval:
            evaluate_agent(model)
    
    elif args.eval:
        print(f"Loading model from {args.model_path}...")
        model = DQN.load(args.model_path)
        evaluate_agent(model)
    
    else:
        print("Please specify --train and/or --eval")
        parser.print_help()


if __name__ == "__main__":
    main()
