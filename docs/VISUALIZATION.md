# Visualization & Replay System

RL Arena provides a comprehensive visualization and replay recording system for analyzing and sharing your RL matches.

## Features

- **Real-time Rendering**: Display games with Matplotlib as they run
- **Frame Recording**: Capture every state for replay analysis
- **JSON Export**: Save matches in a portable format
- **HTML5 Replays**: Interactive browser-based playback with controls
- **Multiple Render Modes**: human, gb_array, nsi, ipython, html

---

## Quick Start

### 1. Basic Visualization

`python
import rl_arena

env = rl_arena.make('pong')
observations, info = env.reset()

for _ in range(100):
    actions = [env.action_space.sample(), env.action_space.sample()]
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    # Render in Matplotlib window
    env.render(mode='human')
    
    if terminated or truncated:
        break

env.close()
`

### 2. Record a Match

`python
from rl_arena.core.recorder import MatchRecorder

# Create recorder with metadata
recorder = MatchRecorder(metadata={
    'env': 'Pong-v1',
    'player1': 'RandomAgent',
    'player2': 'DQNAgent'
})

recorder.start_recording()

# Run your game
env = rl_arena.make('pong')
observations, info = env.reset()

for _ in range(1000):
    actions = [agent1.act(observations[0]), agent2.act(observations[1])]
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    # Record frame
    state = env._get_render_state()
    recorder.record_frame(state, actions, rewards, info)
    
    if terminated or truncated:
        break

recorder.stop_recording()
recorder.save('my_match.json')
print(f'Recorded {len(recorder)} frames')
`

### 3. Generate HTML Replay

`python
from rl_arena.utils.replay import load_replay, replay_to_html

# Load recorded match
replay = load_replay('my_match.json')

# Generate interactive HTML
html = replay_to_html(
    recording=replay,
    env_name='Pong',
    output_path='my_match.html'
)

print('Open my_match.html in your browser!')
`

---

## Render Modes

RL Arena supports multiple rendering modes:

### human - Matplotlib Window
Real-time display in a matplotlib window.

`python
env.render(mode='human')
`

### gb_array - NumPy Array
Returns RGB frame as numpy array (useful for recording videos).

`python
frame = env.render(mode='rgb_array')  # shape: (height, width, 3)
`

### nsi - Terminal Text
ASCII art rendering for terminal display.

`python
text = env.render(mode='ansi')
print(text)
`

### ipython - Jupyter Inline
Display inline in Jupyter notebooks.

`python
# In Jupyter notebook
env.render(mode='ipython')
`

### html - Interactive Replay
Generate HTML5 animation from recorded states.

`python
env.enable_state_recording(True)
# ... run game ...
html = env.render(mode='html')
`

---

## State Recording

Enable automatic state recording to track game history:

`python
env = rl_arena.make('pong')
env.reset()

# Enable recording
env.enable_state_recording(True)

# Run game (states are automatically recorded)
for _ in range(100):
    actions = [env.action_space.sample(), env.action_space.sample()]
    env.step(actions)

# Get recorded history
history = env.get_state_history()
print(f'Recorded {len(history)} states')

# Clear history
env.clear_state_history()
`

---

## MatchRecorder API

### Creating a Recorder

`python
from rl_arena.core.recorder import MatchRecorder

recorder = MatchRecorder(metadata={
    'tournament': 'RL Arena Championship 2025',
    'round': 'semifinals',
    'player1': 'AlphaBot',
    'player2': 'BetaBot'
})
`

### Recording Flow

`python
# Start recording
recorder.start_recording()

# Record frames
recorder.record_frame(
    state={'ball': {'x': 0.5, 'y': 0.5}, ...},
    actions=[0, 1],
    rewards=[1.0, -1.0],
    info={'scores': [5, 3]}
)

# Stop recording
recorder.stop_recording()
`

### Saving and Loading

`python
# Save to JSON
recorder.save('match.json')

# Load from JSON
loaded_recorder = MatchRecorder.load('match.json')
print(f'Loaded {len(loaded_recorder)} frames')
`

### Getting Recording Data

`python
recording = recorder.get_recording()

print(recording['metadata'])
print(recording['num_frames'])
print(recording['duration'])  # in seconds
print(recording['frames'][0])  # first frame
`

---

## Replay Utilities

### Save and Load

`python
from rl_arena.utils.replay import save_replay, load_replay

# Save
recording = recorder.get_recording()
save_replay(recording, 'match.json', pretty=True)

# Load
replay = load_replay('match.json')
`

### Statistics

`python
from rl_arena.utils.replay import get_replay_stats

stats = get_replay_stats(replay)

print(f'Duration: {stats["duration"]} seconds')
print(f'Frames: {stats["num_frames"]}')
print(f'Total rewards: {stats["total_rewards"]}')
print(f'Average rewards: {stats["avg_rewards"]}')
`

### Merge Replays

`python
from rl_arena.utils.replay import merge_replays

replay1 = load_replay('match1.json')
replay2 = load_replay('match2.json')

combined = merge_replays([replay1, replay2])
save_replay(combined, 'tournament.json')
`

### Extract Frames

`python
from rl_arena.utils.replay import extract_frames

# Get first 100 frames
beginning = extract_frames(replay, start=0, end=100)

# Get every 10th frame
sparse = extract_frames(replay, step=10)

# Get last 50 frames
ending = extract_frames(replay, start=-50)
`

---

## HTML5 Replays

### Features

- **Play/Pause Controls**: Start/stop animation
- **Speed Control**: 0.5x, 1x, 2x playback speed
- **Frame Scrubbing**: Drag slider to any frame
- **Keyboard Shortcuts**:
  - Space: Play/Pause
  - `: Previous frame
  - `: Next frame

### Generation

`python
from rl_arena.utils.replay import replay_to_html

html = replay_to_html(
    recording=replay,
    env_name='Pong',
    output_path='match.html'
)
`

### Customization

The HTML template can be customized by modifying l_arena/utils/html_template.py.

---

## Custom Renderers

Create custom renderers for new environments:

`python
from rl_arena.core.renderer import Renderer
import numpy as np

class MyRenderer(Renderer):
    def render_frame(self, state: dict) -> np.ndarray:
        # Return RGB array (height, width, 3)
        pass
    
    def render_html(self, history: list) -> str:
        # Return HTML string
        pass
`

Register with your environment:

`python
class MyEnvironment(Environment):
    def _create_renderer(self) -> Renderer:
        return MyRenderer(width=800, height=600)
    
    def _get_render_state(self) -> dict:
        return {
            'player1': self.player1_pos,
            'player2': self.player2_pos,
            # ... other state info
        }
`

---

## Performance Tips

1. **Disable rendering during training**:
   `python
   # Don't call env.render() in training loops
   `

2. **Use rgb_array for video recording**:
   `python
   frames = []
   for _ in range(1000):
       env.step(actions)
       frames.append(env.render(mode='rgb_array'))
   `

3. **Limit state recording**:
   `python
   # Only enable when needed
   env.enable_state_recording(False)  # Training
   env.enable_state_recording(True)   # Evaluation
   `

4. **Extract sparse replays**:
   `python
   # Save storage with every Nth frame
   sparse = extract_frames(replay, step=5)
   `

---

## Examples

See the examples/ directory for complete examples:

- examples/visualize_game.py - Real-time visualization
- examples/record_match.py - Record and save matches
- examples/replay_to_html.py - Generate HTML replays

---

## Troubleshooting

### Matplotlib not showing window

`python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
`

### HTML replay not animating

- Check browser console for JavaScript errors
- Ensure JSON data is properly formatted
- Try opening in a different browser

### Large file sizes

- Use extract_frames(replay, step=N) to reduce frames
- Save without pretty formatting: save_replay(replay, path, pretty=False)

---

## API Reference

See the [API documentation](api_reference.md) for detailed class and method descriptions.
