# UV Environment Setup

This project uses [UV](https://github.com/astral-sh/uv) for fast Python package management and virtual environment handling.

## Quick Start

1. **Set up the environment:**
   ```bash
   ./setup_uv_env.sh
   ```

2. **Activate the environment:**
   ```bash
   source activate_env.sh
   ```

3. **Or use Make commands:**
   ```bash
   make setup    # Initial setup
   make install  # Install dependencies
   make dev      # Install dev dependencies
   ```

## UV Commands

### Environment Management
- `uv venv` - Create virtual environment
- `uv sync` - Sync dependencies from pyproject.toml
- `uv sync --dev` - Include development dependencies

### Package Management
- `uv add package_name` - Add a package
- `uv add --dev package_name` - Add development package
- `uv remove package_name` - Remove a package
- `uv pip list` - List installed packages

### Running Commands
- `uv run python script.py` - Run Python script in venv
- `uv run pytest` - Run tests
- `uv run black .` - Format code

## Make Commands

We provide a Makefile for common tasks:

```bash
make help      # Show available commands
make setup     # Set up UV environment
make install   # Install dependencies
make dev       # Install dev dependencies
make clean     # Clean environment
make test      # Run tests
make format    # Format code
make lint      # Run linting
make check     # Type checking
make run-train # Start training
make run-eval  # Start evaluation
```

## Environment Variables

Environment variables are loaded from `.env` file automatically when using the activation script.

Required variables:
- `GITHUB_TOKEN` - GitHub personal access token
- `HF_TOKEN` - Hugging Face token

## File Structure

- `pyproject.toml` - Project dependencies and configuration
- `setup_uv_env.sh` - UV installation and setup script
- `activate_env.sh` - Quick environment activation
- `Makefile` - Common development commands
- `.env` - Environment variables (not tracked in git)
