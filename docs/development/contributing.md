# Contributing Guide

Thank you for your interest in contributing to Python AI Core! This guide will help you get started with development.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/example/python-ai-core.git
cd python-ai-core
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We follow these coding standards:
- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](http://mypy-lang.org/) for type checking
- [pylint](https://www.pylint.org/) for code analysis

Our pre-commit hooks will automatically check these when you commit.

## Testing

1. Run the test suite:
```bash
pytest
```

2. Run with coverage:
```bash
pytest --cov=python_ai_core
```

3. Run specific test files:
```bash
pytest tests/test_validation.py
```

## Pull Request Process

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: add new feature"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request on GitHub

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Documentation

1. Build the documentation locally:
```bash
mkdocs serve
```

2. View at http://localhost:8000

3. Update API documentation when adding new features

## Code Review Process

1. All code must be reviewed by at least one maintainer
2. Tests must pass and coverage must not decrease
3. Documentation must be updated if needed
4. Code style checks must pass

## Development Workflow

1. **Pick an Issue**
   - Look for issues labeled `good first issue`
   - Comment that you're working on it

2. **Discuss the Solution**
   - Open a discussion if needed
   - Get feedback on your approach

3. **Write Tests First**
   - Follow Test-Driven Development
   - Ensure good test coverage

4. **Implement the Feature**
   - Keep changes focused
   - Follow code style guidelines

5. **Update Documentation**
   - Add docstrings
   - Update relevant guides

6. **Submit for Review**
   - Create a detailed PR description
   - Link related issues

## Getting Help

- Join our developer chat on Discord
- Ask questions in GitHub Discussions
- Attend our monthly contributor calls

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. 