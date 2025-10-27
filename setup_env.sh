#!/bin/bash

# ===============================
# QCNN Project Environment Setup
# ===============================
# This script sets up a Python environment for running the Quantum Convolutional Neural Network (QCNN) project.
# You can use it to quickly setup dependencies on a new or more powerful machine.
# Run this script from your project's root directory.

echo "=== QCNN Setup Script Starting ==="

# Recommended: Create a fresh virtual environment (requires python3-venv)
echo "making .venv folder if it doesnt exist"
mkdir -p .venv
python3 -m venv .venv
source .venv/bin/activate.fish

# Upgrade pip
pip install --upgrade pip

# --- CORE PYTHON LIBRARIES ---
pip install numpy scipy matplotlib scikit-learn

# --- PENNYLANE AND FAST SIMULATORS ---
pip install pennylane
pip install pennylane-lightning

# --- OTHER DEPENDENCIES ---
# (Add more as needed for your project)
# pip install jupyter  # Optional, for notebook debugging

# --- VERIFY INSTALLATION ---
echo "--- Installed PennyLane devices ---"
python -c "import pennylane as qml; print(qml.list_available_devices())"

echo "=== QCNN Setup Complete! ==="
echo "To activate the environment next time, run: source qcnn_env/bin/activate"
