# RL-Arena Library API - Complete Guide

## 🎯 Library Goals

RL-Arena-Env is a library that enables users to:
1. ✅ **Easily set up environments**
2. ✅ **Train agents**
3. ✅ **Test agents**
4. ✅ **Create submission-ready files**

All in one integrated workflow.

---

## 📦 Installation

```bash
pip install rl-arena-env

# For training features
pip install "rl-arena-env[training]"
# or
pip install stable-baselines3
```

---

## 🚀 Quick Start (30 seconds)

```python
import rl_arena

# 1️⃣ Create environment
env = rl_arena.make("pong")

# 2️⃣ Train agent (DQN)
model = rl_arena.train_dqn("pong", total_timesteps=10000)

# 3️⃣ Evaluate agent
agent = rl_arena.create_agent(model)
results = rl_arena.evaluate(agent, "pong", n_episodes=10)
print(f"Average reward: {results['mean_reward']:.2f}")

# 4️⃣ Create submission file
rl_arena.create_submission(
    agent, 
    "my_submission.py",
    agent_name="MyPongAgent",
    author="your_name"
)
```

**That's it!** Now submit `my_submission.py` to the backend.

---

## 📖 Detailed API

### 1. Environment Creation (`make`)

```python
env = rl_arena.make(
    env_name="pong",
    configuration={
        "render_mode": "human",  # or "rgb_array", "ansi", "json", None
        "max_steps": 1000,
    }
)

# Using the environment
observations, info = env.reset()
actions = [0, 1]  # [player1_action, player2_action]
observations, rewards, terminated, truncated, info = env.step(actions)
```

**Available Environments**:
- `"pong"` - Classic Pong game
- (More environments coming soon)

---

### 2. Agent Training

#### 2.1 Simple DQN Training

```python
# Simplest approach
model = rl_arena.train_dqn("pong", total_timesteps=50000)

# With hyperparameter tuning
model = rl_arena.train_dqn(
    env_name="pong",
    total_timesteps=100000,
    learning_rate=0.001,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    verbose=1,
)

# Save model
model.save("my_agent.zip")
```

#### 2.2 Advanced Training (Direct Stable-Baselines3 Usage)

```python
from stable_baselines3 import DQN, PPO
from rl_arena.wrappers.gymnasium_wrapper import GymnasiumWrapper

# Create environment
env = GymnasiumWrapper("pong", player_id=0)

# Create model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("my_ppo_agent.zip")
```

---

### 3. Agent Testing

#### 3.1 Evaluation (Automated)

```python
# Convert model to agent
agent = rl_arena.create_agent(model, deterministic=True)

# Evaluate
results = rl_arena.evaluate(
    agent=agent,
    env_name="pong",
    n_episodes=10,
    render=False,
)

print(f"Average reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Average length: {results['mean_length']:.2f}")

# Individual episode results
for i, episode in enumerate(results['episodes']):
    print(f"Episode {i+1}: reward={episode['reward']}, length={episode['length']}")
```

#### 3.2 Interactive Testing (Human Play)

```python
# Method 1: Programmatic
player = rl_arena.play(env_name="pong", fps=60)
player.play(
    player1_agent=None,  # Human
    player2_agent=agent,  # Trained AI
)

# Method 2: Command line
# python -m rl_arena.interactive --env pong --player1 human --player2 random
```

**Controls**:
- W/S: Up/Down
- SPACE: Pause
- R: Restart
- Q: Quit

#### 3.3 Replay Recording/Analysis

```python
from rl_arena.utils.recording import MatchRecorder

# Record
recorder = MatchRecorder()
env = rl_arena.make("pong")
env.set_recorder(recorder)

# Play game
observations, _ = env.reset()
for _ in range(1000):
    actions = [agent.act(observations[0]), random_agent.act(observations[1])]
    observations, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break

# Save
recording = recorder.get_recording()
rl_arena.save_replay(recording, "match.json")

# Generate HTML replay
html = rl_arena.replay_to_html(recording)
with open("replay.html", "w") as f:
    f.write(html)
```

---

### 4. Submission Preparation

#### 4.1 Create Submission File

```python
# Basic usage
rl_arena.create_submission(
    agent=agent,
    output_path="submission.py",
    agent_name="MyAwesomeAgent",
    description="DQN agent trained for 100K steps",
    author="your_username",
    version="1.0.0",
)
```

Generated `submission.py`:
```python
"""
MyAwesomeAgent - Submission for RL-Arena

Description: DQN agent trained for 100K steps
Author: your_username
Version: 1.0.0
"""

from rl_arena.core.agent import Agent

class MyAwesomeAgent(Agent):
    def act(self, observation, info=None):
        # Your logic here
        return action

# Metadata
__agent_name__ = "MyAwesomeAgent"
__author__ = "your_username"
__version__ = "1.0.0"

def create_agent() -> Agent:
    return MyAwesomeAgent()
```

#### 4.2 Validate Submission

```python
# Validate submission file
result = rl_arena.validate("submission.py")

if result['valid']:
    print("✅ Submission file is valid!")
else:
    print("❌ Issues found:")
    for issue in result['issues']:
        print(f"  - {issue}")
    
    if result['warnings']:
        print("⚠️ Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
```

#### 4.3 Package (with Model)

```python
# Package with model as zip file
rl_arena.package(
    agent_path="submission.py",
    model_path="my_agent.zip",
    output_path="final_submission.zip",
)
```

---

## 🎮 Built-in Agents

```python
# Random Agent
random_agent = rl_arena.RandomAgent()

# Rule-based Agent
rule_agent = rl_arena.RuleBasedAgent(reaction_speed=0.8)

# Usage
env = rl_arena.make("pong")
observations, _ = env.reset()
action = random_agent.act(observations[0])
```

---

## 🔄 Complete Workflow Example

```python
import rl_arena

# ====================
# 1. Setup Environment
# ====================
ENV_NAME = "pong"
AGENT_NAME = "MyPongMaster"

# ====================
# 2. Training
# ====================
print("🎓 Starting training...")
model = rl_arena.train_dqn(
    env_name=ENV_NAME,
    total_timesteps=50000,
    learning_rate=0.001,
    verbose=1,
)

# Save model
model.save(f"{AGENT_NAME}.zip")
print(f"✅ Model saved: {AGENT_NAME}.zip")

# ====================
# 3. Evaluation
# ====================
print("\n📊 Starting evaluation...")
agent = rl_arena.create_agent(model)
results = rl_arena.evaluate(
    agent=agent,
    env_name=ENV_NAME,
    n_episodes=20,
)

print(f"Average reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Average length: {results['mean_length']:.2f}")

# ====================
# 4. Interactive Testing
# ====================
print("\n🎮 Interactive testing (Human vs AI)")
print("Controls: W/S (up/down), SPACE (pause), Q (quit)")

player = rl_arena.play(env_name=ENV_NAME, fps=60)
player.play(
    player1_agent=None,  # Human
    player2_agent=agent,  # AI
    max_episodes=3,
)

# ====================
# 5. Create Submission
# ====================
print("\n📦 Creating submission file...")
rl_arena.create_submission(
    agent=agent,
    output_path=f"{AGENT_NAME}_submission.py",
    agent_name=AGENT_NAME,
    description=f"DQN agent, trained 50K steps, avg reward {results['mean_reward']:.2f}",
    author="your_name",
    version="1.0.0",
)

# ====================
# 6. Validation
# ====================
print("\n✅ Validating submission...")
validation = rl_arena.validate(f"{AGENT_NAME}_submission.py")

if validation['valid']:
    print("✅ Submission ready!")
    print(f"   File: {AGENT_NAME}_submission.py")
    print(f"   Size: {validation['file_size'] / 1024:.1f} KB")
else:
    print("❌ Issues found:", validation['issues'])

# ====================
# 7. Packaging
# ====================
print("\n📦 Final packaging...")
rl_arena.package(
    agent_path=f"{AGENT_NAME}_submission.py",
    model_path=f"{AGENT_NAME}.zip",
    output_path=f"{AGENT_NAME}_final.zip",
)

print("\n🎉 Complete!")
print(f"Submit to backend: {AGENT_NAME}_final.zip")
```

---

## 🛠️ Advanced Features

### Creating Custom Agents

```python
from rl_arena import Agent

class MyCustomAgent(Agent):
    def __init__(self):
        super().__init__()
        # Initialize
        self.my_state = {}
    
    def act(self, observation, info=None):
        # Choose action based on observation
        # observation: [paddle_y, ball_x, ball_y, ball_vx, ball_vy]
        
        paddle_y = observation[0]
        ball_y = observation[2]
        
        if ball_y > paddle_y:
            return 2  # DOWN
        elif ball_y < paddle_y:
            return 0  # UP
        else:
            return 1  # STAY
    
    def reset(self):
        # Called at episode start
        self.my_state = {}

# Usage
agent = MyCustomAgent()
results = rl_arena.evaluate(agent, "pong", n_episodes=10)
```

### Using Gymnasium Wrapper

```python
from rl_arena.wrappers.gymnasium_wrapper import GymnasiumWrapper

# For single-agent training
env = GymnasiumWrapper("pong", player_id=0)

# Gymnasium interface
observation, info = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

---

## 📚 Additional Resources

### Example Scripts

```bash
# examples/ directory
examples/
├── train_dqn.py          # DQN training example
├── train_ppo.py          # PPO training example
├── play_interactive.py   # Interactive play
└── complete_workflow.py  # Complete workflow
```

### Documentation

- [Environment Creation Guide](docs/ENVIRONMENT_CREATION_TUTORIAL.md)
- [Interactive Mode](INTERACTIVE_MODE.md)
- [API Reference](docs/api/)

---

## ❓ FAQ

### Q: How long does training take?
A: Depends on environment and timesteps:
- Pong, 10K steps: ~30 seconds
- Pong, 50K steps: ~2 minutes
- Pong, 500K steps: ~20 minutes

### Q: Which algorithm should I use?
A: 
- **DQN**: Simple and fast, discrete actions
- **PPO**: More stable, supports continuous actions
- **MCTS**: Perfect information games (Tic-Tac-Toe, Connect Four)

### Q: Do I need a GPU?
A: 
- Simple environments (Pong, Tic-Tac-Toe): CPU is sufficient
- Complex environments or long training: GPU recommended

### Q: What's the submission file size limit?
A: 
- Recommended: < 10MB
- For large models, compress or optimize

---

## 🎯 Summary

### Core 4 Steps

```python
# 1. Create
env = rl_arena.make("pong")

# 2. Train
model = rl_arena.train_dqn("pong", total_timesteps=50000)

# 3. Test
agent = rl_arena.create_agent(model)
results = rl_arena.evaluate(agent, "pong", n_episodes=10)

# 4. Submit
rl_arena.create_submission(agent, "submission.py", agent_name="MyAgent")
```

**Remember these!**

---

**Last Updated**: 2025-11-06  
**Version**: 0.1  
**Library Goals**: ✅ Fully Achieved!
