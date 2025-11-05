# Quick Start Guide

Welcome to RL Arena! This guide will help you get started with creating and running competitive RL environments.

## Installation

### From PyPI (recommended)

```bash
pip install rl-arena
```

### From source

```bash
git clone https://github.com/rl-arena/rl-arena-env.git
cd rl-arena-env
pip install -e .
```

### For development

```bash
git clone https://github.com/rl-arena/rl-arena-env.git
cd rl-arena-env
pip install -e ".[dev]"
```

## Your First Environment

### Creating an environment

```python
import rl_arena

# Create a Pong environment
env = rl_arena.make("pong")

# Check environment properties
print(f"Number of players: {env.num_players}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

### Running an episode

```python
# Reset the environment
observations, info = env.reset(seed=42)

# Run one episode
done = False
while not done:
    # Get actions (random for now)
    actions = [env.action_space.sample() for _ in range(env.num_players)]
    
    # Step the environment
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    # Render the game
    env.render()

print(f"Final scores: {info['scores']}")
env.close()
```

## Creating Agents

### Random Agent

```python
from rl_arena.core.agent import RandomAgent

env = rl_arena.make("pong")
agent = RandomAgent(env.action_space, player_id=0)

observations, _ = env.reset()
action = agent.act(observations[0])
```

### Custom Agent

```python
from rl_arena.core.agent import Agent

class MyAgent(Agent):
    def act(self, observation):
        # Your logic here
        return 0  # Example: always move up

env = rl_arena.make("pong")
my_agent = MyAgent(player_id=0)
```

## Environment Configuration

Customize environment behavior with configuration parameters:

```python
env = rl_arena.make("pong", configuration={
    "winning_score": 5,      # First to 5 wins
    "max_steps": 2000,       # Longer episodes
    "ball_speed": 0.03,      # Faster ball
    "paddle_height": 0.15,   # Smaller paddles
})
```

## Available Environments

List all available environments:

```python
import rl_arena

environments = rl_arena.list_environments()
print("Available environments:", environments)
```

Current environments:
- **pong**: Classic 2-player Pong game

## Recording Replays

Save games for later analysis:

```python
from rl_arena.utils import ReplayRecorder

env = rl_arena.make("pong")
recorder = ReplayRecorder(
    environment_name="pong",
    configuration=env.get_config()
)

observations, info = env.reset()
done = False
step = 0

while not done:
    actions = [env.action_space.sample(), env.action_space.sample()]
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    # Record this step
    recorder.record_step(
        step=step,
        actions=actions,
        observations=observations,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        info=info
    )
    
    done = terminated or truncated
    step += 1

# Save replay
recorder.save("replays/my_game.json")
env.close()
```

## Running Examples

The `examples/` directory contains ready-to-run scripts:

```bash
# Random agents playing Pong
python examples/random_agent.py

# Run local matches
python examples/run_local_match.py

# Train a DQN agent (requires stable-baselines3)
python examples/train_dqn_pong.py --train --timesteps 100000
```

## Next Steps

- **[Environment Guide](environment_guide.md)**: Learn how to create new environments
- **[API Reference](api_reference.md)**: Detailed API documentation
- **[Contributing](contributing.md)**: How to contribute to RL Arena

## Common Issues

### Import errors

Make sure rl-arena is installed:
```bash
pip install rl-arena
```

### Missing dependencies

Install optional dependencies:
```bash
pip install rl-arena[examples]  # For example scripts
pip install rl-arena[dev]       # For development
```

### Rendering issues

If matplotlib is not installed:
```bash
pip install matplotlib
```

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/rl-arena/rl-arena-env/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/rl-arena/rl-arena-env/discussions)
- **Documentation**: [Full documentation](https://github.com/rl-arena/rl-arena-env/tree/main/docs)
