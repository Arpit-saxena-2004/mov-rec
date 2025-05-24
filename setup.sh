#!/bin/bash
echo "Installing torch CPU version first..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Setup completed."