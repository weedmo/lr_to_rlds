#!/bin/bash
# LeRobot to RLDS Converter - Environment Setup Script
# Usage: source setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "LeRobot to RLDS Converter - Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "[1/5] Creating virtual environment..."
    python3 -m venv .venv
else
    echo ""
    echo "[1/5] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[2/5] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip wheel setuptools -q

# Initialize and update rlds submodule
echo ""
echo "[4/5] Setting up rlds submodule..."
if [ ! -f "rlds/setup.py" ] && [ ! -f "rlds/pip_package/setup.py" ]; then
    git submodule update --init --recursive
fi

# Install rlds submodule
if [ -f "rlds/pip_package/setup.py" ]; then
    pip install -e ./rlds/pip_package -q
elif [ -f "rlds/setup.py" ]; then
    pip install -e ./rlds -q
else
    echo "Warning: rlds submodule not found, installing from PyPI..."
    pip install rlds -q
fi

# Install main package with dev dependencies
echo ""
echo "[5/5] Installing lerobot-to-rlds..."
pip install -e ".[dev]" -q

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  # Activate environment (if not already active)"
echo "  source .venv/bin/activate"
echo ""
echo "  # Convert dataset (OXE/OpenVLA format - default)"
echo "  lerobot-to-rlds convert /path/to/dataset --output ./output"
echo ""
echo "  # Discover dataset info"
echo "  lerobot-to-rlds discover /path/to/dataset"
echo ""
echo "  # Run tests"
echo "  pytest"
echo ""
