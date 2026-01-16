#!/bin/bash
# Quick start script for speaker-diarization-community-1-mlx

echo "=========================================="
echo "Speaker Diarization MLX - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install mlx numpy scipy scikit-learn librosa soundfile --quiet

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run tests: python tests/test_basic.py"
echo "  3. Try examples: python examples/basic_usage.py"
echo ""
echo "To convert a model:"
echo "  python src/convert.py --pytorch-path MODEL_PATH --mlx-path models/mlx_model"
echo ""
