@echo off
REM ===============================
REM QCNN Project Environment Setup (Windows Batch)
REM ===============================
REM This script sets up a Python environment for running the Quantum Convolutional Neural Network (QCNN) project.
REM Run this script from your project's root directory.

echo === QCNN Setup Script Starting ===

REM Recommended: Create a fresh virtual environment
echo Making .venv folder if it doesn't exist
if not exist .venv (
    python -m venv .venv
) else (
    echo .venv already exists.
)

REM Activate the virtual environment
echo Activating .venv...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM --- CORE PYTHON LIBRARIES ---
echo Installing core libraries...
pip install numpy scipy matplotlib scikit-learn

REM --- PENNYLANE AND FAST SIMULATORS ---
echo Installing PennyLane...
pip install pennylane
pip install pennylane-lightning

REM --- OTHER DEPENDENCIES ---
REM (Add more as needed for your project)
REM pip install jupyter

REM --- VERIFY INSTALLATION ---
echo --- Installed PennyLane devices ---
python -c "import pennylane as qml; print(qml.list_available_devices())"

echo === QCNN Setup Complete! ===
echo To activate the environment next time, run: .venv\Scripts\activate
pause
