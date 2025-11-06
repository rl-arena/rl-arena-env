"""Example: Convert a recorded match to HTML5 replay.

This script demonstrates how to:
1. Load a saved match recording
2. Convert it to an interactive HTML5 animation
3. Save the HTML file for viewing in a browser
"""

from rl_arena.utils.replay import load_replay, replay_to_html, get_replay_stats
from pathlib import Path
import webbrowser


def main():
    # Check if recording exists
    recording_file = Path("recordings/pong_match.json")

    if not recording_file.exists():
        print("Error: Recording file not found!")
        print("Please run 'record_match.py' first to create a recording.")
        return

    print(f"Loading recording from: {recording_file}")

    # Load the recording
    recording = load_replay(str(recording_file))

    # Display recording info
    print(f"\nRecording Information:")
    print(f"  Frames: {recording['num_frames']}")
    print(f"  Duration: {recording.get('duration', 'N/A')} seconds")

    stats = get_replay_stats(recording)
    print(f"  Total rewards: {stats['total_rewards']}")

    # Convert to HTML
    print("\nGenerating HTML5 replay...")
    output_file = Path("recordings/pong_replay.html")

    html = replay_to_html(recording=recording, env_name="Pong", output_path=str(output_file))

    print(f"Saved HTML replay to: {output_file}")
    print(f"\nHTML file size: {len(html)} characters")

    # Ask if user wants to open in browser
    response = input("\nOpen replay in browser? (y/n): ").strip().lower()
    if response == "y":
        print("Opening in default browser...")
        webbrowser.open(f"file://{output_file.absolute()}")
    else:
        print(f"\nYou can open the file manually: {output_file.absolute()}")

    print("\nControls in the HTML replay:")
    print("  - Play/Pause buttons")
    print("  - Frame slider for scrubbing")
    print("  - Speed controls (0.5x, 1x, 2x)")
    print("  - Keyboard: Space=play/pause, Arrow keys=frame by frame")


if __name__ == "__main__":
    main()
