"""Example: Simple random agent playing Pong.

This example demonstrates the basic usage of RL Arena with random agents.
Both players take random actions at each step.
"""

import rl_arena
from rl_arena.core.agent import RandomAgent


def main():
    """Run a Pong game with two random agents."""
    # Create the environment
    env = rl_arena.make("pong")

    # Create two random agents
    agent1 = RandomAgent(env.action_space, player_id=0)
    agent2 = RandomAgent(env.action_space, player_id=1)

    # Reset the environment
    observations, info = env.reset(seed=42)
    print("Starting Pong game with random agents...")
    print(f"Initial scores: {info['scores']}")

    # Run one episode
    episode_rewards = [[], []]
    done = False
    step = 0

    while not done:
        # Get actions from both agents
        action1 = agent1.act(observations[0])
        action2 = agent2.act(observations[1])
        actions = [action1, action2]

        # Step the environment
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Track rewards
        episode_rewards[0].append(rewards[0])
        episode_rewards[1].append(rewards[1])

        # Render (optional - comment out for faster execution)
        env.render(mode="ansi")

        step += 1

        # Print score updates
        if any(r != 0 for r in rewards):
            print(f"Step {step}: Player scores: {info['scores']}")

    # Print final results
    print("\n" + "=" * 50)
    print("Game Over!")
    print(f"Final scores: Player 1: {info['scores'][0]}, Player 2: {info['scores'][1]}")
    print(f"Total steps: {step}")
    print(
        f"Total rewards - Player 1: {sum(episode_rewards[0])}, Player 2: {sum(episode_rewards[1])}"
    )

    # Determine winner
    if info["scores"][0] > info["scores"][1]:
        print("Winner: Player 1")
    elif info["scores"][1] > info["scores"][0]:
        print("Winner: Player 2")
    else:
        print("It's a draw!")

    env.close()


if __name__ == "__main__":
    main()
