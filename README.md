# RL Arena 🎮

A Python library for competitive reinforcement learning environments, similar to kaggle-environments but focused on multi-agent RL research and competitions.

[![Tests](https://github.com/rl-arena/rl-arena-env/workflows/Tests/badge.svg)](https://github.com/rl-arena/rl-arena-env/actions)
[![PyPI version](https://badge.fury.io/py/rl-arena.svg)](https://badge.fury.io/py/rl-arena)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Features

- **Easy-to-use API**: Familiar Gymnasium-style interface
- **Competitive Multi-Agent**: Built for head-to-head competition
- **Extensible**: Simple framework for adding new environments
- **Well-tested**: Comprehensive test suite with >80% coverage
- **Type-safe**: Full type hints for better development experience
- **Reproducible**: Deterministic environments with seed support
- **Replay System**: Save and analyze game replays

## 📦 Installation

### From PyPI (recommended)

```bash
pip install rl-arena
```

### From Source

```bash
git clone https://github.com/rl-arena/rl-arena-env.git
cd rl-arena-env
pip install -e .
```

### For Development

```bash
git clone https://github.com/rl-arena/rl-arena-env.git
cd rl-arena-env
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
import rl_arena
from rl_arena.core.agent import RandomAgent

# Create environment
env = rl_arena.make("pong")

# Create agents
agent1 = RandomAgent(env.action_space, player_id=0)
agent2 = RandomAgent(env.action_space, player_id=1)

# Run a game
observations, info = env.reset(seed=42)
done = False

while not done:
    # Get actions from agents
    actions = [
        agent1.act(observations[0]),
        agent2.act(observations[1])
    ]
    
    # Step environment
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    # Render
    env.render()

print(f"Final scores: {info['scores']}")
env.close()
```

## 🎯 Available Environments

| Environment | Players | Description | Complexity |
|-------------|---------|-------------|------------|
| **pong** | 2 | Classic Pong game | ⭐ Easy |

**More environments coming soon!** Contributions welcome! 

## 🎮 Environment Details

### Pong

Classic 2-player Pong game with customizable physics.

```python
env = rl_arena.make("pong", configuration={
    "winning_score": 11,    # First to 11 wins
    "max_steps": 1000,      # Episode length limit
    "ball_speed": 0.02,     # Ball movement speed
    "paddle_height": 0.2,   # Paddle size
})
```

**State Space:** Ball position & velocity, paddle positions, scores  
**Action Space:** Discrete(3) - UP, STAY, DOWN  
**Rewards:** +1 for scoring, -1 for conceding

See [environment documentation](rl_arena/envs/pong/README.md) for more details.

## 📚 Documentation

- **[Quick Start](docs/quickstart.md)**: Get started in 5 minutes
- **[Environment Guide](docs/environment_guide.md)**: Create custom environments
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Contributing](docs/contributing.md)**: Contribution guidelines

## 🔨 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rl_arena tests/

# Run specific test file
pytest tests/test_core.py -v
```

### Code Quality

```bash
# Format code
black rl_arena/ tests/ examples/

# Type checking
mypy rl_arena/

# Linting
flake8 rl_arena/ tests/ examples/
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add New Environments**: See [Environment Guide](docs/environment_guide.md)
2. **Fix Bugs**: Check [open issues](https://github.com/rl-arena/rl-arena-env/issues)
3. **Improve Documentation**: Help make RL Arena more accessible
4. **Add Examples**: Share your agent implementations

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

### Quick Contribution Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/rl-arena-env.git
cd rl-arena-env

# Create branch
git checkout -b feature/my-new-environment

# Make changes, add tests, update docs

# Run tests and linting
pytest
black rl_arena/ tests/
flake8 rl_arena/

# Commit and push
git commit -m "feat: Add new environment"
git push origin feature/my-new-environment

# Open Pull Request on GitHub
```

## 📖 Examples

### Random Agents

```bash
python examples/random_agent.py
```

### Train a DQN Agent

```bash
pip install stable-baselines3[extra]
python examples/train_dqn_pong.py --train --timesteps 100000
```

### Run Local Matches

```bash
python examples/run_local_match.py
```

### Agent Submission Template

```bash
python examples/submission_template.py
```

## 🎓 Use Cases

- **Research**: Study multi-agent RL algorithms
- **Education**: Learn RL in competitive settings
- **Competitions**: Host RL tournaments and competitions
- **Benchmarking**: Compare agent performance across environments
- **Fun**: Build and battle AI agents!

## 🏗️ Project Structure

```
rl-arena-env/
├── rl_arena/              # Main package
│   ├── core/              # Base classes and interfaces
│   ├── envs/              # Environment implementations
│   │   ├── pong/          # Pong environment
│   │   └── __template__/  # Template for new environments
│   └── utils/             # Utilities (replay, logging, etc.)
├── tests/                 # Test suite
├── examples/              # Example scripts
├── docs/                  # Documentation
└── .github/               # CI/CD workflows
```



## 🙏 Acknowledgments

- Inspired by [kaggle-environments](https://github.com/Kaggle/kaggle-environments)
- Built on [Gymnasium](https://gymnasium.farama.org/)
- Thanks to all [contributors](https://github.com/rl-arena/rl-arena-env/graphs/contributors)

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/rl-arena/rl-arena-env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rl-arena/rl-arena-env/discussions)


## ⭐ Star History

If you find RL Arena useful, please consider starring the repository!

---

**Made with ❤️ by the RL Arena community**