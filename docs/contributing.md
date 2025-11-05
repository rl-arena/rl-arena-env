# Contributing to RL Arena

Thank you for your interest in contributing to RL Arena! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Adding New Environments](#adding-new-environments)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

Be respectful, inclusive, and constructive. We want RL Arena to be a welcoming community for everyone.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, rl-arena version)
- **Code sample** if applicable

Use the bug report template when creating issues.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear, descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this would be useful** to most users
- **List any alternatives** you've considered

### Contributing Code

#### Types of Contributions

1. **New Environments**: Add competitive RL environments
2. **Bug Fixes**: Fix issues in existing code
3. **Features**: Add new functionality
4. **Documentation**: Improve docs, add examples
5. **Tests**: Increase test coverage

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/rl-arena-env.git
cd rl-arena-env
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/issue-123
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if you changed APIs
2. **Add tests** for new functionality
3. **Run tests** and ensure they pass
4. **Run linters** (black, mypy, flake8)
5. **Update CHANGELOG** if applicable

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=rl_arena tests/
```

### Running Linters

```bash
# Format code with black
black rl_arena/ tests/ examples/

# Check types with mypy
mypy rl_arena/

# Check style with flake8
flake8 rl_arena/ tests/ examples/
```

### Submitting the PR

1. **Push your branch** to your fork
2. **Open a Pull Request** against `main`
3. **Fill out the PR template** completely
4. **Link any related issues**
5. **Wait for review** and address feedback

### PR Title Format

Use conventional commit format:

- `feat: Add Connect4 environment`
- `fix: Fix Pong ball collision detection`
- `docs: Update quickstart guide`
- `test: Add tests for registry`
- `refactor: Simplify environment base class`

## Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (black default)
- **Imports**: Grouped and sorted (stdlib, third-party, local)
- **Type hints**: Use for all public functions
- **Docstrings**: Google style

Example:

```python
from typing import List, Tuple, Optional
import numpy as np

def calculate_reward(
    player_id: int,
    game_state: np.ndarray,
    action: int
) -> float:
    """
    Calculate reward for a player's action.
    
    Args:
        player_id: ID of the player (0-indexed)
        game_state: Current state of the game
        action: Action taken by the player
        
    Returns:
        Reward value (positive for beneficial actions)
        
    Example:
        >>> reward = calculate_reward(0, state, 2)
        >>> print(reward)
        1.0
    """
    # Implementation...
    return 0.0
```

### Documentation Style

- **Classes**: Describe purpose and usage
- **Methods**: Include Args, Returns, Raises, Example
- **Modules**: Add module-level docstring
- **README**: Keep updated with new features

### Commit Messages

Write clear, concise commit messages:

```
feat: Add replay saving functionality

- Implement ReplayRecorder class
- Add save/load replay functions
- Update documentation with examples
```

## Adding New Environments

See [Environment Guide](environment_guide.md) for detailed instructions.

### Quick Checklist

- [ ] Copy and customize template from `rl_arena/envs/__template__/`
- [ ] Implement all required methods
- [ ] Add configuration file (`config.yaml`)
- [ ] Write comprehensive README
- [ ] Add tests in `tests/envs/test_your_env.py`
- [ ] Create example script in `examples/`
- [ ] Register environment in `rl_arena/envs/registry.py`
- [ ] Run all tests and linters
- [ ] Update main README with new environment

### Environment Requirements

Your environment must:

1. **Inherit from `Environment`** base class
2. **Implement all abstract methods**
3. **Use `@register_environment` decorator**
4. **Have proper type hints**
5. **Include comprehensive docstrings**
6. **Be deterministic** with same seed
7. **Handle edge cases** gracefully
8. **Have tests** with >80% coverage

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_core.py          # Core functionality tests
├── test_registry.py      # Registry tests
├── test_make.py          # Factory function tests
└── envs/
    └── test_pong.py      # Environment-specific tests
```

### Writing Tests

Use pytest with descriptive test names:

```python
def test_environment_reset_returns_correct_types():
    """Test that reset returns observations and info dict."""
    env = rl_arena.make("pong")
    observations, info = env.reset()
    
    assert isinstance(observations, list)
    assert len(observations) == 2
    assert isinstance(info, dict)
    
    env.close()
```

### Test Coverage

Aim for >80% coverage:

```bash
pytest --cov=rl_arena --cov-report=html tests/
# View report: open htmlcov/index.html
```

## Documentation

### Types of Documentation

1. **Docstrings**: In-code documentation
2. **README.md**: Project overview, installation, quick start
3. **docs/*.md**: Detailed guides
4. **examples/**: Runnable code examples

### Building Documentation

Documentation is in Markdown format in the `docs/` directory.

### Documentation Checklist

- [ ] All public classes/functions have docstrings
- [ ] Docstrings include examples
- [ ] README is updated
- [ ] Relevant guide is updated
- [ ] Example scripts work correctly

## Community

### Getting Help

- **GitHub Discussions**: Ask questions, share ideas
- **GitHub Issues**: Report bugs, request features
- **Pull Requests**: Contribute code, docs, tests

### Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- GitHub contributors graph

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open a GitHub Discussion
- Comment on related issues
- Ask in your Pull Request

Thank you for contributing to RL Arena! 🎮🤖
