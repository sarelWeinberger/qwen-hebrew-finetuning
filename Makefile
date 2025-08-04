.PHONY: setup install dev clean test format lint check run-train run-eval help

# Default Python version
PYTHON_VERSION := 3.10

# Help command
help:
	@echo "Available commands:"
	@echo "  setup     - Install UV and set up virtual environment"
	@echo "  install   - Install project dependencies"
	@echo "  dev       - Install development dependencies"
	@echo "  clean     - Clean virtual environment and cache"
	@echo "  test      - Run tests"
	@echo "  format    - Format code with black"
	@echo "  lint      - Run linting"
	@echo "  check     - Run type checking"
	@echo "  run-train - Run training pipeline"
	@echo "  run-eval  - Run evaluation"

# Setup UV and virtual environment
setup:
	@echo "🚀 Setting up UV environment..."
	@./setup_uv_env.sh

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@uv sync

# Install development dependencies
dev:
	@echo "📦 Installing development dependencies..."
	@uv sync --dev

# Clean environment
clean:
	@echo "🧹 Cleaning environment..."
	@rm -rf .venv
	@rm -rf .uv
	@rm -rf __pycache__
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +

# Run tests
test:
	@echo "🧪 Running tests..."
	@uv run pytest

# Format code
format:
	@echo "🎨 Formatting code..."
	@uv run black .

# Run linting
lint:
	@echo "🔍 Running linting..."
	@uv run flake8 .

# Run type checking
check:
	@echo "🔍 Running type checking..."
	@uv run mypy .

# Training commands
run-train:
	@echo "🏋️ Starting training..."
	@uv run python qwen_model/main.py

run-eval:
	@echo "📊 Starting evaluation..."
	@uv run python qwen_model/evaluate_hebrew.py

# Environment info
info:
	@echo "Environment Information:"
	@echo "Python: $$(uv run python --version)"
	@echo "UV: $$(uv --version)"
	@echo "Virtual Environment: $$(echo $$VIRTUAL_ENV)"
