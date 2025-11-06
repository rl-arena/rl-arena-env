"""Example: Record a Pong match and save to file.

This script demonstrates how to:
1. Use MatchRecorder to record gameplay
2. Save recordings to JSON format
3. Extract and display match statistics
"""

from rl_arena.envs.pong.environment import PongEnvironment
from rl_arena.core.recorder import MatchRecorder
from rl_arena.utils.replay import get_replay_stats
from pathlib import Path


def main():
    # Create environment
    env = PongEnvironment(configuration={"winning_score": 3, "max_steps": 500})

    # Create recorder with metadata
    recorder = MatchRecorder(
        metadata={
            "env": "Pong-v1",
            "player1": "RandomAgent",
            "player2": "RandomAgent",
            "description": "Demo match recording",
        }
    )

    # Start recording
    recorder.start_recording()
    print("Recording match...")

    # Reset environment
    observations, info = env.reset(seed=42)

    # Game loop
    step = 0
    done = False

    while not done:
        # Random actions
        actions = [env.action_space.sample(), env.action_space.sample()]

        # Step environment
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Record frame with current state
        state = {
            "ball": {"x": float(env.ball_pos[0]), "y": float(env.ball_pos[1])},
            "paddle1": {"y": float(env.paddle1_y)},
            "paddle2": {"y": float(env.paddle2_y)},
            "paddle_height": env.paddle_height,
            "score": [env.score1, env.score2],
        }
        recorder.record_frame(state, actions, rewards, info)

        step += 1

        if sum(rewards) != 0:
            print(f"Step {step}: Scores = {info['scores']}")

    # Stop recording
    recorder.stop_recording()
    print(f"\nRecording complete! Captured {len(recorder)} frames")

    # Save recording
    output_dir = Path("recordings")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "pong_match.json"

    recorder.save(str(output_file))
    print(f"Saved recording to: {output_file}")

    # Display statistics
    recording = recorder.get_recording()
    stats = get_replay_stats(recording)

    print("\nMatch Statistics:")
    print(f"  Duration: {stats['duration']:.2f} seconds")
    print(f"  Total frames: {stats['num_frames']}")
    print(f"  Total rewards: {stats['total_rewards']}")
    print(f"  Average rewards: {[f'{r:.2f}' for r in stats['avg_rewards']]}")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
