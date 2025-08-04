#!/bin/bash

# Quick activation script for the UV virtual environment
# Usage: source activate_env.sh

if [ -f ".venv/bin/activate" ]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
    
    # Load environment variables
    if [ -f ".env" ]; then
        echo "🔧 Loading environment variables..."
        set -a
        source .env
        set +a
    fi
    
    echo "✅ Environment activated!"
    echo "Python: $(which python)"
    echo "UV: $(which uv)"
else
    echo "❌ Virtual environment not found. Run ./setup_uv_env.sh first"
fi
