# ===============================
# QCNN Project Environment Setup (PowerShell)
# ===============================
# This script sets up a Python environment for running the Quantum Convolutional Neural Network (QCNN) project.
# Run this script from your project's root directory.

Write-Host "=== QCNN Setup Script Starting ===" -ForegroundColor Cyan

# Recommended: Create a fresh virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Making .venv folder..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists."
}

# Activate the virtual environment
Write-Host "Activating .venv..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# --- CORE PYTHON LIBRARIES ---
Write-Host "Installing core libraries..." -ForegroundColor Yellow
pip install numpy scipy matplotlib scikit-learn

# --- PENNYLANE AND FAST SIMULATORS ---
Write-Host "Installing PennyLane..." -ForegroundColor Yellow
pip install pennylane
pip install pennylane-lightning

# --- OTHER DEPENDENCIES ---
# (Add more as needed for your project)
# pip install jupyter

# --- VERIFY INSTALLATION ---
Write-Host "--- Installed PennyLane devices ---" -ForegroundColor Green
python -c "import pennylane as qml; print(qml.list_available_devices())"

Write-Host "=== QCNN Setup Complete! ===" -ForegroundColor Cyan
Write-Host "To activate the environment next time, run: .\.venv\Scripts\Activate.ps1"
Pause
