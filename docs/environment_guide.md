# Environment Creation Guide

This guide walks you through creating a new environment for RL Arena. By following these steps, you can add new competitive games that others can use for RL research and competitions.

## Overview

Creating a new environment involves:
1. Implementing the `Environment` base class
2. Registering your environment
3. Adding configuration and documentation
4. Writing tests
5. Submitting a pull request

## Step 1: Use the Template

Start with the environment template in `rl_arena/envs/__template__/`:

```bash
cd rl_arena/envs
cp -r __template__ my_game
cd my_game
```

Rename the template files:
```bash
mv __init__.py.template __init__.py
mv environment.py.template environment.py
mv README.md.template README.md
```

## Step 2: Implement Your Environment

### Define the Environment Class

Edit `environment.py` and implement the required methods:

```python
from rl_arena.core.environment import Environment
from rl_arena.envs.registry import register_environment
import gymnasium as gym
import numpy as np

@register_environment("my-game")  # Unique name (lowercase, hyphenated)
class MyGameEnvironment(Environment):
    """
    Your game description here.
    """
    
    def __init__(self, configuration=None):
        super().__init__(configuration)
        # Initialize your game state
        self._num_players = 2
        
    @property
    def num_players(self):
        return self._num_players
    
    @property
    def action_space(self):
        # Define action space (e.g., Discrete, Box, MultiDiscrete)
        return gym.spaces.Discrete(9)  # Example: 9 possible actions
    
    @property
    def observation_space(self):
        # Define observation space
        return gym.spaces.Box(
            low=0, high=1, 
            shape=(8, 8),  # Example: 8x8 board
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        # Reset game to initial state
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize game state
        self._current_step = 0
        self._done = False
        
        # Generate initial observations
        observations = [self._get_observation(i) for i in range(self.num_players)]
        info = {}
        
        return observations, info
    
    def step(self, actions):
        # Apply actions and update game state
        self._current_step += 1
        
        # Update state based on actions
        # ...
        
        # Calculate rewards
        rewards = [0.0 for _ in range(self.num_players)]
        
        # Check termination
        terminated = False  # Game ended naturally
        truncated = False   # Episode cut off
        
        # Generate observations
        observations = [self._get_observation(i) for i in range(self.num_players)]
        info = {}
        
        return observations, rewards, terminated, truncated, info
    
    def render(self, mode="human"):
        # Implement rendering
        if mode == "ansi":
            return self._render_ansi()
        # ...
    
    def get_observation(self, player_id):
        # Return observation for specific player
        return self._get_observation(player_id)
    
    def _get_observation(self, player_id):
        # Helper method to generate observations
        # ...
        pass
```

### Key Implementation Guidelines

#### Action Space
Choose the appropriate action space type:
- **Discrete(n)**: For n discrete actions (e.g., chess moves)
- **Box**: For continuous actions
- **MultiDiscrete**: For multiple discrete choices

#### Observation Space
Make observations informative but compact:
- Include all information players need to make decisions
- For partially observable games, use `get_observation(player_id)`
- Normalize values to reasonable ranges (e.g., [0, 1])

#### Rewards
Design rewards carefully:
- Sparse rewards: Only at game end (simpler but harder to learn)
- Dense rewards: At each step (easier to learn but more design effort)
- Zero-sum: One player's gain is another's loss (common in competitive games)

#### Episode Termination
Set clear termination conditions:
- **terminated**: Natural end (win/loss/draw)
- **truncated**: Artificial end (time limit, max steps)

## Step 3: Add Configuration

Create `config.yaml` for default settings:

```yaml
# Default configuration for MyGame

# Game parameters
board_size: 8
max_steps: 100

# Rendering
render_fps: 30
```

Load it in your `__init__`:

```python
from pathlib import Path
import yaml

config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as f:
    default_config = yaml.safe_load(f)

self.config = {**default_config, **self.configuration}
```

## Step 4: Write Documentation

Create a comprehensive `README.md` in your environment directory:

```markdown
# My Game Environment

## Overview
Brief description of the game.

## State Space
Detailed description of observations.

## Action Space
Detailed description of actions.

## Rewards
Explanation of reward structure.

## Episode Termination
When does an episode end?

## Configuration Options
Table of configuration parameters.

## Example Usage
Code example showing how to use your environment.
```

## Step 5: Write Tests

Create `tests/envs/test_my_game.py`:

```python
import pytest
import rl_arena
from rl_arena.core.exceptions import InvalidActionError

def test_my_game_creation():
    """Test environment creation."""
    env = rl_arena.make("my-game")
    assert env.num_players == 2
    env.close()

def test_my_game_reset():
    """Test reset functionality."""
    env = rl_arena.make("my-game")
    observations, info = env.reset(seed=42)
    assert len(observations) == 2
    env.close()

def test_my_game_step():
    """Test step functionality."""
    env = rl_arena.make("my-game")
    env.reset()
    
    actions = [env.action_space.sample() for _ in range(2)]
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    assert len(observations) == 2
    assert len(rewards) == 2
    env.close()

# Add more tests...
```

Run tests:
```bash
pytest tests/envs/test_my_game.py -v
```

## Step 6: Register Your Environment

Update `rl_arena/envs/registry.py` to import your environment:

```python
# Import environments to trigger registration
from rl_arena.envs import pong
from rl_arena.envs import my_game  # Add this line
```

## Step 7: Add Example Script

Create `examples/my_game_example.py`:

```python
import rl_arena
from rl_arena.core.agent import RandomAgent

def main():
    env = rl_arena.make("my-game")
    agents = [RandomAgent(env.action_space, i) for i in range(2)]
    
    observations, _ = env.reset(seed=42)
    done = False
    
    while not done:
        actions = [agent.act(obs) for agent, obs in zip(agents, observations)]
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        env.render()
    
    env.close()

if __name__ == "__main__":
    main()
```

## Checklist Before Submission

- [ ] Environment implements all required methods
- [ ] Environment is registered with a unique name
- [ ] Configuration file exists (if needed)
- [ ] README.md is complete with examples
- [ ] Tests are written and passing
- [ ] Example script is included
- [ ] Code follows style guidelines (black, mypy)
- [ ] Documentation is clear and comprehensive

## Testing Your Environment

Before submitting, test thoroughly:

```python
import rl_arena

# Test basic functionality
env = rl_arena.make("my-game")
observations, _ = env.reset(seed=42)

for _ in range(100):
    actions = [env.action_space.sample() for _ in range(2)]
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    if terminated or truncated:
        break

# Test determinism
obs1, _ = env.reset(seed=42)
obs2, _ = env.reset(seed=42)
assert (obs1[0] == obs2[0]).all()

# Test configuration
env2 = rl_arena.make("my-game", configuration={"board_size": 10})
assert env2.config["board_size"] == 10

env.close()
```

## Submission Process

1. **Fork** the repository
2. **Create a branch** for your environment: `git checkout -b add-my-game`
3. **Implement** your environment
4. **Test** thoroughly
5. **Commit** with clear messages
6. **Push** to your fork
7. **Open a Pull Request** using the environment addition template

See [CONTRIBUTING.md](contributing.md) for detailed submission guidelines.

## Examples to Learn From

Study existing environments for reference:
- **Pong**: Simple physics-based game (`rl_arena/envs/pong/`)
- More environments coming soon!

## Getting Help

- **Template**: Use `rl_arena/envs/__template__/` as a starting point
- **GitHub Discussions**: Ask questions about environment design
- **Issues**: Report problems or request clarifications

## Best Practices

1. **Keep it simple**: Start with a minimal working version
2. **Test thoroughly**: Edge cases, invalid actions, episode boundaries
3. **Document well**: Clear docstrings and README
4. **Make it configurable**: Use configuration files for flexibility
5. **Follow conventions**: Match the style of existing environments
6. **Add examples**: Help users understand your environment

## Advanced Topics

### Partial Observability

For games where players don't see everything:

```python
def get_observation(self, player_id):
    # Return player-specific view
    if player_id == 0:
        return self._get_player_0_view()
    else:
        return self._get_player_1_view()
```

### Stochastic Environments

Use seeding properly for reproducibility:

```python
def reset(self, seed=None, options=None):
    if seed is not None:
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
```

### Complex Renderings

Implement multiple rendering modes:

```python
def render(self, mode="human"):
    if mode == "human":
        return self._render_matplotlib()
    elif mode == "rgb_array":
        return self._render_rgb()
    elif mode == "ansi":
        return self._render_ansi()
```

## FAQ

**Q: How do I handle different numbers of players?**
A: Set `self._num_players` in `__init__` and ensure all methods handle that many players.

**Q: Should I include AI opponents in my environment?**
A: No, environments should just define the game mechanics. Users will provide their own agents.

**Q: How do I handle variable-length episodes?**
A: Use `terminated` for natural endings and `truncated` for artificial limits (max_steps).

**Q: Can I use external dependencies?**
A: Core environments should use only numpy and gymnasium. Optional dependencies (like pygame) should be imported conditionally.
