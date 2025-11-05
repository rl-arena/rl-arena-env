# API Reference

Complete API documentation for RL Arena.

## Core Module (`rl_arena.core`)

### Environment

```python
class Environment(ABC)
```

Abstract base class for all RL Arena environments.

#### Properties

- **`num_players: int`** - Number of players in the environment
- **`action_space: gym.Space`** - Action space for the environment
- **`observation_space: gym.Space`** - Observation space for the environment
- **`current_step: int`** - Current step number
- **`is_done: bool`** - Whether the episode has ended

#### Methods

##### `reset(seed=None, options=None)`

Reset the environment to initial state.

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `options` (dict, optional): Additional reset options

**Returns:**
- `observations` (list): List of initial observations for each player
- `info` (dict): Additional information

**Example:**
```python
observations, info = env.reset(seed=42)
```

##### `step(actions)`

Execute one step in the environment.

**Parameters:**
- `actions` (list): List of actions, one per player

**Returns:**
- `observations` (list): Observations for each player
- `rewards` (list): Rewards for each player
- `terminated` (bool): Whether episode ended naturally
- `truncated` (bool): Whether episode was cut off
- `info` (dict): Additional information

**Example:**
```python
observations, rewards, terminated, truncated, info = env.step([action1, action2])
```

##### `render(mode='human')`

Render the environment.

**Parameters:**
- `mode` (str): Rendering mode
  - `'human'`: Display to screen
  - `'rgb_array'`: Return RGB array
  - `'ansi'`: Return ASCII string

**Returns:**
- Depends on mode (None, numpy array, or string)

**Example:**
```python
env.render(mode='ansi')
```

##### `get_observation(player_id)`

Get observation for a specific player.

**Parameters:**
- `player_id` (int): Player ID (0-indexed)

**Returns:**
- Observation for the specified player

**Raises:**
- `InvalidPlayerError`: If player_id is invalid

**Example:**
```python
obs = env.get_observation(0)
```

##### `close()`

Clean up environment resources.

**Example:**
```python
env.close()
```

##### `get_config()`

Get current configuration.

**Returns:**
- `config` (dict): Configuration dictionary

---

### Agent

```python
class Agent(ABC)
```

Abstract base class for agents.

#### Methods

##### `act(observation)`

Choose an action based on observation.

**Parameters:**
- `observation`: Current observation

**Returns:**
- `action`: Action to take

**Example:**
```python
action = agent.act(observation)
```

##### `reset()`

Reset agent's internal state.

**Example:**
```python
agent.reset()
```

---

### RandomAgent

```python
class RandomAgent(Agent)
```

Agent that selects random actions.

**Parameters:**
- `action_space` (gym.Space): Action space
- `player_id` (int): Player ID

**Example:**
```python
from rl_arena.core.agent import RandomAgent

agent = RandomAgent(env.action_space, player_id=0)
action = agent.act(observation)
```

---

## Factory Function

### `make(env_name, configuration=None)`

Create an environment by name.

**Parameters:**
- `env_name` (str): Name of the environment
- `configuration` (dict, optional): Configuration parameters

**Returns:**
- `env` (Environment): Environment instance

**Raises:**
- `EnvironmentNotFoundError`: If environment not found

**Example:**
```python
import rl_arena

env = rl_arena.make("pong")
env = rl_arena.make("pong", configuration={"winning_score": 5})
```

### `list_environments()`

List all registered environments.

**Returns:**
- `environments` (list): Sorted list of environment names

**Example:**
```python
envs = rl_arena.list_environments()
print(envs)  # ['pong', ...]
```

---

## Exceptions (`rl_arena.core.exceptions`)

### `RLArenaException`

Base exception for all RL Arena errors.

### `EnvironmentError`

Raised for environment-related errors.

### `EnvironmentNotFoundError`

Raised when environment is not found.

**Example:**
```python
try:
    env = rl_arena.make("nonexistent")
except EnvironmentNotFoundError as e:
    print(f"Error: {e}")
```

### `InvalidActionError`

Raised when an invalid action is taken.

**Attributes:**
- `action`: The invalid action
- `player_id`: ID of the player who took the action

### `InvalidConfigurationError`

Raised when configuration is invalid.

### `GameEndedError`

Raised when trying to step after episode ended.

### `InvalidPlayerError`

Raised when an invalid player ID is used.

---

## Utilities (`rl_arena.utils`)

### ReplayRecorder

Record environment interactions.

```python
class ReplayRecorder
```

**Parameters:**
- `environment_name` (str): Name of environment
- `configuration` (dict, optional): Environment configuration
- `metadata` (dict, optional): Additional metadata

**Methods:**

#### `record_step(step, actions, observations, rewards, terminated, truncated, info)`

Record a single step.

#### `save(filepath)`

Save replay to JSON file.

**Example:**
```python
from rl_arena.utils import ReplayRecorder

recorder = ReplayRecorder("pong")
# ... record steps ...
recorder.save("replay.json")
```

### `load_replay(filepath)`

Load a replay from file.

**Parameters:**
- `filepath` (str): Path to replay file

**Returns:**
- `replay_data` (dict): Replay data

**Example:**
```python
from rl_arena.utils import load_replay

replay = load_replay("replay.json")
print(replay['metadata'])
```

---

### Validation

#### `validate_action(action, action_space, player_id)`

Validate an action.

**Raises:**
- `InvalidActionError`: If action is invalid

**Example:**
```python
from rl_arena.utils import validate_action

validate_action(action, env.action_space, player_id=0)
```

#### `validate_configuration(configuration, required_keys, optional_keys=None, key_types=None)`

Validate configuration dictionary.

**Raises:**
- `InvalidConfigurationError`: If configuration is invalid

**Example:**
```python
from rl_arena.utils import validate_configuration

validate_configuration(
    config,
    required_keys=["max_steps"],
    optional_keys=["ball_speed"],
    key_types={"max_steps": int}
)
```

---

### Logging

#### `setup_logging(level=logging.INFO, log_file=None)`

Set up logging for RL Arena.

**Parameters:**
- `level` (int): Logging level
- `log_file` (str, optional): File to write logs to

**Example:**
```python
from rl_arena.utils import setup_logging
import logging

setup_logging(level=logging.DEBUG, log_file="logs/rl_arena.log")
```

#### `get_logger(name)`

Get a logger instance.

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logger` (logging.Logger): Logger instance

**Example:**
```python
from rl_arena.utils import get_logger

logger = get_logger(__name__)
logger.info("Training started")
```

---

## Environment Registry

### `register_environment(name)`

Decorator to register an environment.

**Parameters:**
- `name` (str): Name to register under

**Returns:**
- Decorator function

**Example:**
```python
from rl_arena.envs.registry import register_environment
from rl_arena.core.environment import Environment

@register_environment("my-game")
class MyGameEnvironment(Environment):
    # Implementation...
    pass
```

### `get_environment_class(name)`

Get environment class by name.

**Parameters:**
- `name` (str): Environment name

**Returns:**
- Environment class

**Raises:**
- `KeyError`: If environment not found

---

## Type Hints

```python
from rl_arena.core.types import (
    ObservationType,
    ActionType,
    RewardType,
    InfoType,
    ConfigType,
)
```

- **`ObservationType`**: Union[np.ndarray, Dict, int, float]
- **`ActionType`**: Union[int, np.ndarray, Dict]
- **`RewardType`**: Union[int, float]
- **`InfoType`**: Dict[str, Any]
- **`ConfigType`**: Dict[str, Any]

---

## Version

```python
import rl_arena

print(rl_arena.__version__)  # '0.1.0'
```

---

## Complete Example

```python
import rl_arena
from rl_arena.core.agent import RandomAgent
from rl_arena.utils import ReplayRecorder, setup_logging
import logging

# Setup logging
setup_logging(level=logging.INFO)

# Create environment
env = rl_arena.make("pong", configuration={
    "winning_score": 5,
    "max_steps": 1000
})

# Create agents
agents = [
    RandomAgent(env.action_space, player_id=0),
    RandomAgent(env.action_space, player_id=1)
]

# Setup replay recorder
recorder = ReplayRecorder("pong", configuration=env.get_config())

# Run episode
observations, info = env.reset(seed=42)
done = False
step = 0

while not done:
    # Get actions
    actions = [agent.act(obs) for agent, obs in zip(agents, observations)]
    
    # Step environment
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    # Record step
    recorder.record_step(step, actions, observations, rewards, 
                        terminated, truncated, info)
    
    # Render
    if step % 10 == 0:
        env.render()
    
    step += 1

# Save replay
recorder.save("replays/game.json")

# Cleanup
env.close()

print(f"Game finished after {step} steps")
print(f"Final scores: {info['scores']}")
```
