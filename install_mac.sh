#!/usr/bin/env bash
set -e

# Vai nella cartella dove si trova questo script
cd "$(dirname "$0")"

echo "Creating virtual environment (venv)..."
python3 -m venv venv

echo "Activating venv..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo
echo "✅ Installation complete."
echo "To run the simulator, use: ./run_mac.sh"
