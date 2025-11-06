# Creating a New Environment for RL-Arena

This tutorial will guide you through creating a new competitive RL environment for RL-Arena. We'll build a simple **Coin Collection** game as an example.

## Table of Contents

1. [Overview](#overview)
2. [Game Design](#game-design)
3. [Step 1: Scaffold the Environment](#step-1-scaffold-the-environment)
4. [Step 2: Define Spaces](#step-2-define-spaces)
5. [Step 3: Implement Game Logic](#step-3-implement-game-logic)
6. [Step 4: Implement Rendering](#step-4-implement-rendering)
7. [Step 5: Write Tests](#step-5-write-tests)
8. [Step 6: Validate and Polish](#step-6-validate-and-polish)
9. [Step 7: Register Environment](#step-7-register-environment)

---

## Overview

Creating a new RL-Arena environment involves:

1. **Scaffolding** - Generate template files
2. **Spaces** - Define observation and action spaces
3. **Game Logic** - Implement reset() and step()
4. **Rendering** - Visualize the game
5. **Testing** - Ensure correctness
6. **Validation** - Verify API compliance
7. **Registration** - Make it available to users

---

## Game Design

### Coin Collection

**Concept**: Two players compete to collect coins on a grid.

**Rules**:
- 5x5 grid world
- 3 coins spawn at random positions
- Players can move: UP, DOWN, LEFT, RIGHT, STAY
- Collecting a coin gives +1 point
- First to 2 coins wins
- If all coins collected, player with more wins

**Why this game?**:
- Simple to implement
- Turn-based (no complex physics)
- Good for testing AI strategies
- Teaches core concepts

---

## Step 1: Scaffold the Environment

Use the environment creation script:

```bash
python scripts/create_environment.py "Coin Collection"
```

This creates:
```
rl_arena/envs/coin_collection/
├── __init__.py
├── environment.py
├── renderer.py
├── test_coin_collection.py
└── README.md
```

---

## Step 2: Define Spaces

Edit `rl_arena/envs/coin_collection/environment.py`:

### Action Space

```python
def _create_action_space(self) -> gym.Space:
    """Create the action space for the environment."""
    # 5 discrete actions: UP, DOWN, LEFT, RIGHT, STAY
    return gym.spaces.Discrete(5)
```

### Observation Space

```python
def _create_observation_space(self) -> gym.Space:
    """Create the observation space for the environment."""
    # Observation: [player1_x, player1_y, player2_x, player2_y, 
    #               coin1_x, coin1_y, coin2_x, coin2_y, coin3_x, coin3_y,
    #               score1, score2]
    # All normalized to [0, 1]
    return gym.spaces.Box(
        low=0.0, 
        high=1.0, 
        shape=(12,), 
        dtype=np.float32
    )
```

**Why this design?**:
- Simple flat vector (easy for neural networks)
- Normalized values (better for training)
- Contains all necessary information

---

## Step 3: Implement Game Logic

### State Initialization

```python
def _reset_state(self) -> None:
    """Reset the internal game state."""
    # Grid size
    self.grid_size = 5
    
    # Player positions (x, y)
    self.player_positions = [
        [0, 0],  # Player 1 starts at top-left
        [4, 4],  # Player 2 starts at bottom-right
    ]
    
    # Scores
    self.scores = [0, 0]
    
    # Spawn coins at random positions
    self._spawn_coins()
    
def _spawn_coins(self) -> None:
    """Spawn coins at random empty positions."""
    self.coins = []
    while len(self.coins) < 3:
        x = self._np_random.integers(0, self.grid_size)
        y = self._np_random.integers(0, self.grid_size)
        pos = [x, y]
        
        # Check if position is empty
        if (pos not in self.player_positions and 
            pos not in self.coins):
            self.coins.append(pos)
```

### Reset Method

```python
def reset(
    self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
) -> Tuple[List[ObservationType], Dict[str, Any]]:
    """Reset the environment to initial state."""
    # Handle seeding
    super().reset(seed=seed)
    if seed is not None:
        self._seed = seed
        self._np_random = np.random.default_rng(seed)
    elif self._np_random is None:
        self._np_random = np.random.default_rng()
    
    # Reset game state
    self._reset_state()
    self._current_step = 0
    self._done = False
    
    # Clear state history if recording
    if self._record_states:
        self._state_history = []
        self._state_history.append(self._get_render_state())
    
    # Get initial observations
    observations = self._get_observations()
    
    # Info dict
    info = {
        "step": self._current_step,
        "scores": self.scores.copy(),
    }
    
    return observations, info
```

### Step Method

```python
def step(
    self, actions: List[ActionType]
) -> Tuple[List[ObservationType], List[RewardType], bool, bool, Dict[str, Any]]:
    """Execute one step in the environment."""
    # Validate actions
    if len(actions) != self.num_players:
        raise ValueError(f"Expected {self.num_players} actions, got {len(actions)}")
    
    for i, action in enumerate(actions):
        if not self.action_space.contains(action):
            raise ValueError(f"Player {i} took invalid action: {action}")
    
    # Move players
    for i, action in enumerate(actions):
        self._move_player(i, action)
    
    # Check coin collection
    rewards = [0.0, 0.0]
    for i in range(self.num_players):
        coins_collected = self._check_coin_collection(i)
        self.scores[i] += coins_collected
        rewards[i] += coins_collected
    
    # Check win condition
    terminated = False
    if self.scores[0] >= 2 or self.scores[1] >= 2:
        terminated = True
        # Winner gets bonus reward
        if self.scores[0] > self.scores[1]:
            rewards[0] += 1.0
            rewards[1] -= 1.0
        elif self.scores[1] > self.scores[0]:
            rewards[1] += 1.0
            rewards[0] -= 1.0
    
    # Check if all coins collected
    if len(self.coins) == 0 and not terminated:
        terminated = True
        # Winner gets bonus
        if self.scores[0] > self.scores[1]:
            rewards[0] += 1.0
            rewards[1] -= 1.0
        elif self.scores[1] > self.scores[0]:
            rewards[1] += 1.0
            rewards[0] -= 1.0
    
    # Increment step counter
    self._current_step += 1
    
    # Check truncation
    truncated = self._current_step >= self.max_steps
    
    # Get observations
    observations = self._get_observations()
    
    # Record state
    if self._record_states:
        self._state_history.append(self._get_render_state())
    
    # Info dict
    info = {
        "step": self._current_step,
        "scores": self.scores.copy(),
    }
    
    return observations, rewards, terminated, truncated, info

def _move_player(self, player_id: int, action: int) -> None:
    """Move a player based on action."""
    x, y = self.player_positions[player_id]
    
    # Action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
    if action == 0:  # UP
        y = max(0, y - 1)
    elif action == 1:  # DOWN
        y = min(self.grid_size - 1, y + 1)
    elif action == 2:  # LEFT
        x = max(0, x - 1)
    elif action == 3:  # RIGHT
        x = min(self.grid_size - 1, x + 1)
    # action == 4: STAY (no change)
    
    self.player_positions[player_id] = [x, y]

def _check_coin_collection(self, player_id: int) -> int:
    """Check if player collected any coins."""
    pos = self.player_positions[player_id]
    collected = 0
    
    # Check all coins
    coins_to_remove = []
    for i, coin_pos in enumerate(self.coins):
        if pos == coin_pos:
            collected += 1
            coins_to_remove.append(i)
    
    # Remove collected coins
    for i in reversed(coins_to_remove):
        self.coins.pop(i)
    
    return collected
```

### Observations

```python
def _get_observations(self) -> List[ObservationType]:
    """Get current observations for all players."""
    # Normalize positions to [0, 1]
    obs = np.zeros(12, dtype=np.float32)
    
    # Player positions
    obs[0] = self.player_positions[0][0] / (self.grid_size - 1)
    obs[1] = self.player_positions[0][1] / (self.grid_size - 1)
    obs[2] = self.player_positions[1][0] / (self.grid_size - 1)
    obs[3] = self.player_positions[1][1] / (self.grid_size - 1)
    
    # Coin positions (pad with -1 if coin collected)
    for i in range(3):
        if i < len(self.coins):
            obs[4 + i*2] = self.coins[i][0] / (self.grid_size - 1)
            obs[5 + i*2] = self.coins[i][1] / (self.grid_size - 1)
        else:
            obs[4 + i*2] = -1.0  # Coin collected
            obs[5 + i*2] = -1.0
    
    # Scores (normalized)
    obs[10] = self.scores[0] / 3.0
    obs[11] = self.scores[1] / 3.0
    
    # Both players see the same observation
    # (In asymmetric games, you might flip perspectives)
    return [obs.copy(), obs.copy()]
```

---

## Step 4: Implement Rendering

Edit `rl_arena/envs/coin_collection/renderer.py`:

### ASCII Rendering

```python
def render_ansi(self, state: Dict[str, Any]) -> str:
    """Render as ASCII string for terminal."""
    grid_size = state['grid_size']
    player_positions = state['player_positions']
    coins = state['coins']
    scores = state['scores']
    
    # Create grid
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Place coins
    for coin in coins:
        x, y = coin
        grid[y][x] = '$'
    
    # Place players
    for i, pos in enumerate(player_positions):
        x, y = pos
        grid[y][x] = str(i + 1)
    
    # Build string
    lines = []
    lines.append("╔" + "═" * (grid_size * 2 + 1) + "╗")
    for row in grid:
        lines.append("║ " + " ".join(row) + " ║")
    lines.append("╠" + "═" * (grid_size * 2 + 1) + "╣")
    lines.append(f"║ P1: {scores[0]}  |  P2: {scores[1]} " + " " * (grid_size * 2 - 11) + "║")
    lines.append("╚" + "═" * (grid_size * 2 + 1) + "╝")
    
    return "\n".join(lines)
```

### RGB Frame Rendering

```python
def render_frame(self, state: Dict[str, Any]) -> np.ndarray:
    """Render a single frame as RGB array."""
    grid_size = state['grid_size']
    player_positions = state['player_positions']
    coins = state['coins']
    
    # Frame size
    cell_size = 60
    width = grid_size * cell_size
    height = grid_size * cell_size + 40  # +40 for score display
    
    # Create black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw grid
    for i in range(grid_size + 1):
        # Vertical lines
        frame[:height-40, i*cell_size:i*cell_size+1] = [50, 50, 50]
        # Horizontal lines
        frame[i*cell_size:i*cell_size+1, :] = [50, 50, 50]
    
    # Draw coins (yellow circles)
    for coin in coins:
        x, y = coin
        cx = int(x * cell_size + cell_size // 2)
        cy = int(y * cell_size + cell_size // 2)
        self._draw_circle(frame, cx, cy, cell_size // 4, [255, 255, 0])
    
    # Draw players
    colors = [[0, 150, 255], [255, 100, 100]]  # Blue, Red
    for i, pos in enumerate(player_positions):
        x, y = pos
        cx = int(x * cell_size + cell_size // 2)
        cy = int(y * cell_size + cell_size // 2)
        self._draw_circle(frame, cx, cy, cell_size // 3, colors[i])
    
    # Draw scores
    score_area = frame[height-40:height, :]
    score_area[:] = [30, 30, 30]
    # (Add text rendering with PIL or cv2 if available)
    
    return frame

def _draw_circle(self, frame: np.ndarray, cx: int, cy: int, 
                 radius: int, color: List[int]) -> None:
    """Draw a filled circle on the frame."""
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    frame[mask] = color
```

---

## Step 5: Write Tests

Edit `tests/envs/test_coin_collection.py`:

```python
def test_coin_collection_game_logic():
    """Test that coins are collected correctly."""
    env = CoinCollectionEnvironment()
    env.reset(seed=42)
    
    # Manually set up a test scenario
    env.player_positions = [[0, 0], [4, 4]]
    env.coins = [[0, 1], [2, 2], [4, 3]]
    env.scores = [0, 0]
    
    # Player 1 moves down to collect coin
    observations, rewards, terminated, truncated, info = env.step([1, 4])  # DOWN, STAY
    
    assert rewards[0] == 1.0  # Player 1 collected coin
    assert rewards[1] == 0.0  # Player 2 did nothing
    assert env.scores[0] == 1
    assert len(env.coins) == 2  # One coin collected

def test_win_condition():
    """Test that game ends when player reaches 2 coins."""
    env = CoinCollectionEnvironment()
    env.reset(seed=42)
    
    # Set up winning scenario
    env.scores = [1, 0]
    env.player_positions = [[0, 0], [4, 4]]
    env.coins = [[0, 1]]
    
    # Player 1 collects final coin
    observations, rewards, terminated, truncated, info = env.step([1, 4])
    
    assert terminated is True
    assert env.scores[0] == 2
    assert rewards[0] > rewards[1]  # Winner gets bonus
```

---

## Step 6: Validate and Polish

### Run Validator

```python
from rl_arena.tools import validate_environment
from rl_arena.envs.coin_collection import CoinCollectionEnvironment

validate_environment(CoinCollectionEnvironment)
```

### Run Tests

```bash
pytest tests/envs/test_coin_collection.py -v
```

### Common Issues

**Issue**: "Observation not in observation_space"
- **Fix**: Check normalization (values should be [0, 1])

**Issue**: "Episode never terminates"
- **Fix**: Add truncation or ensure win condition is reachable

**Issue**: "Action validation fails"
- **Fix**: Verify action_space.contains() logic

---

## Step 7: Register Environment

Add to `rl_arena/envs/__init__.py`:

```python
from rl_arena.envs.coin_collection import CoinCollectionEnvironment

AVAILABLE_ENVS = {
    "pong": "rl_arena.envs.pong.PongEnvironment",
    "coin_collection": "rl_arena.envs.coin_collection.CoinCollectionEnvironment",
}
```

Add to `rl_arena/core/registry.py`:

```python
def make(env_name: str, configuration: Optional[Dict[str, Any]] = None) -> Environment:
    """Create an environment instance."""
    if env_name == "coin_collection":
        from rl_arena.envs.coin_collection import CoinCollectionEnvironment
        return CoinCollectionEnvironment(configuration)
    # ... other environments
```

---

## Testing Your Environment

### Quick Test

```python
from rl_arena import make

env = make("coin_collection", {"render_mode": "human"})
observations, info = env.reset(seed=42)

for _ in range(100):
    actions = [env.action_space.sample() for _ in range(2)]
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()
    
    if terminated or truncated:
        print(f"Game over! Scores: {info['scores']}")
        break
```

---

## Best Practices

### DO ✅
- Keep game logic simple initially
- Normalize observations to [0, 1]
- Add comprehensive docstrings
- Test edge cases (boundaries, ties)
- Use seeding for reproducibility
- Record states for replay

### DON'T ❌
- Mix rendering with game logic
- Forget to validate actions
- Skip writing tests
- Make observations too complex
- Hardcode magic numbers

---

## Next Steps

1. **Optimize** - Profile and improve performance
2. **Balance** - Tune rewards and difficulty
3. **Visualize** - Improve rendering quality
4. **Document** - Write detailed README
5. **Share** - Submit PR to rl-arena-env
6. **Train Agents** - Test with DQN, PPO, etc.

---

## Resources

- [RL-Arena Documentation](https://github.com/rl-arena/rl-arena-docs)
- [Gymnasium Spaces](https://gymnasium.farama.org/api/spaces/)
- [Example: Pong Environment](../rl_arena/envs/pong/)
- [Environment API Reference](../docs/api_reference.md)

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/rl-arena/rl-arena-env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rl-arena/rl-arena-env/discussions)


