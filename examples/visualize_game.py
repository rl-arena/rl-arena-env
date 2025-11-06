"""Example: Visualize a Pong game with matplotlib rendering.

This script demonstrates how to:
1. Create a Pong environment
2. Run a game with random agents
3. Render the game in real-time using matplotlib
"""

from rl_arena.envs.pong.environment import PongEnvironment
import time


def main():
    # Create environment
    env = PongEnvironment(
        configuration={"winning_score": 5, "max_steps": 500}  # Shorter game for demo
    )

    # Reset environment
    observations, info = env.reset(seed=42)
    print("Starting Pong game visualization...")
    print(f"Initial scores: {info['scores']}")

    # Enable state recording for potential replay later
    env.enable_state_recording(True)

    # Game loop
    step = 0
    done = False

    while not done:
        # Random actions for both players (for demo purposes)
        actions = [env.action_space.sample(), env.action_space.sample()]

        # Step the environment
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Render the game
        env.render(mode="human")

        # Small delay for visualization
        time.sleep(0.016)  # ~60 FPS

        step += 1

        # Print score updates
        if sum(rewards) != 0:
            print(f"Step {step}: Scores = {info['scores']}, Rewards = {rewards}")

    print(f"\nGame ended after {step} steps")
    print(f"Final scores: {info['scores']}")

    # Get state history
    history = env.get_state_history()
    print(f"Recorded {len(history)} frames")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
