"""Template for submitting an agent to RL Arena competitions.

This template shows how to structure your agent submission for
RL Arena competitions. Copy this file and implement your agent logic.
"""

import numpy as np
from typing import Any


class MyAgent:
    """
    Your custom agent for RL Arena competitions.
    
    TODO: Implement your agent's logic in the methods below.
    """
    
    def __init__(self, player_id: int = 0):
        """
        Initialize your agent.
        
        Args:
            player_id: The player ID assigned to this agent (0 or 1)
            
        Note:
            This method is called once when your agent is created.
            Use it to initialize any models, load weights, etc.
        """
        self.player_id = player_id
        
        # TODO: Initialize your agent
        # Examples:
        # - Load a trained model
        # - Initialize neural network
        # - Set up lookup tables
        # - Load configuration
        
        # Example: self.model = load_model("my_agent.h5")
        
    def act(self, observation: Any) -> int:
        """
        Choose an action based on the observation.
        
        This is the main method that will be called at each step.
        
        Args:
            observation: The current observation from the environment.
                        Format depends on the environment (see environment docs).
            
        Returns:
            action: The action to take (integer for discrete actions)
            
        Note:
            This method should be fast! In competitions, there may be
            time limits on how long your agent can take to choose an action.
        """
        # TODO: Implement your agent's decision logic
        
        # Example for Pong:
        # observation = [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y, score1, score2]
        # action = 0 (UP), 1 (STAY), or 2 (DOWN)
        
        # Placeholder: Random action
        action = np.random.randint(0, 3)
        
        return action
    
    def reset(self) -> None:
        """
        Reset your agent's internal state (if any).
        
        This method is called at the start of each episode.
        Use it to reset any episodic state.
        
        Note:
            This method is optional. Implement it only if your agent
            maintains internal state across steps.
        """
        # TODO: Reset any episodic state
        # Examples:
        # - Clear history buffers
        # - Reset RNN hidden states
        # - Clear cached predictions
        pass
    
    # Optional: Add any helper methods you need
    # def _preprocess_observation(self, observation):
    #     """Preprocess the observation before feeding to your model."""
    #     pass
    # 
    # def _get_features(self, observation):
    #     """Extract features from the observation."""
    #     pass


# ============================================================================
# Testing Your Agent Locally
# ============================================================================

def test_agent():
    """
    Test your agent locally before submission.
    
    This function shows how to test your agent in a local environment.
    Make sure your agent works correctly before submitting!
    """
    import rl_arena
    from rl_arena.core.agent import RandomAgent
    
    # Create environment
    env = rl_arena.make("pong")  # Change to your target environment
    
    # Create agents
    my_agent = MyAgent(player_id=0)
    opponent = RandomAgent(env.action_space, player_id=1)
    
    # Run test episodes
    num_episodes = 5
    wins = 0
    
    for episode in range(num_episodes):
        observations, info = env.reset(seed=episode)
        my_agent.reset()
        
        done = False
        episode_reward = 0
        
        while not done:
            # Get actions
            my_action = my_agent.act(observations[0])
            opponent_action = opponent.act(observations[1])
            
            # Step environment
            observations, rewards, terminated, truncated, info = env.step(
                [my_action, opponent_action]
            )
            done = terminated or truncated
            
            episode_reward += rewards[0]
        
        # Check result
        if info['scores'][0] > info['scores'][1]:
            wins += 1
            result = "WIN"
        elif info['scores'][0] < info['scores'][1]:
            result = "LOSS"
        else:
            result = "DRAW"
        
        print(f"Episode {episode + 1}: {result} - "
              f"Scores: {info['scores']}, Reward: {episode_reward:.1f}")
    
    print(f"\nResults: {wins}/{num_episodes} wins ({wins/num_episodes:.1%} win rate)")
    
    env.close()


if __name__ == "__main__":
    print("Testing your agent locally...")
    print("="*60)
    test_agent()
    print("="*60)
    print("\nIf your agent works well, you're ready to submit!")
    print("Follow the submission guidelines at: https://rl-arena.dev/submit")
