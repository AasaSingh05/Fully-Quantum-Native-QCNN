@echo off
setlocal enabledelayedexpansion

:: Windows Launcher for QCNN Training
:: Usage: runApp.bat [type] [path] [encoding] [image_size] [class1 class2]

:: Handle flags passed as first argument
set "first_arg=%~1"
if "!first_arg:~0,2!"=="--" (
    echo Detected flags. Using default dataset (MNIST/auto)...
    python main.py --dataset auto --path datasets/MNIST --encoding auto --image-size 28 --classes 0 1 --no-profile %*
    exit /b %errorlevel%
)

:: Configuration with defaults
set "DATASET_TYPE=%~1"
if "%DATASET_TYPE%"=="" set "DATASET_TYPE=auto"

set "DATASET_PATH=%~2"
if "%DATASET_PATH%"=="" set "DATASET_PATH=datasets/MNIST"

set "ENCODING=%~3"
if "%ENCODING%"=="" set "ENCODING=auto"

set "IMAGE_SIZE=%~4"
if "%IMAGE_SIZE%"=="" set "IMAGE_SIZE=28"

set "CLASS_A=%~5"
if "%CLASS_A%"=="" set "CLASS_A=0"

set "CLASS_B=%~6"
if "%CLASS_B%"=="" set "CLASS_B=1"

:: Shift arguments to pass remaining flags to python
shift
shift
shift
shift
shift
shift

echo ------------------------------------------------
echo Quantum Native QCNN - Training Launcher (Win)
echo ------------------------------------------------
echo Dataset: %DATASET_PATH% (%DATASET_TYPE%)
echo Encoding: %ENCODING% (Size: %IMAGE_SIZE%)
echo Classes: %CLASS_A% vs %CLASS_B%
echo ------------------------------------------------

:: Check for virtual environment
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py ^
        --dataset "%DATASET_TYPE%" ^
        --path "%DATASET_PATH%" ^
        --encoding "%ENCODING%" ^
        --image-size "%IMAGE_SIZE%" ^
        --classes "%CLASS_A%" "%CLASS_B%" ^
        --no-profile ^
        %*
) else (
    python main.py ^
        --dataset "%DATASET_TYPE%" ^
        --path "%DATASET_PATH%" ^
        --encoding "%ENCODING%" ^
        --image-size "%IMAGE_SIZE%" ^
        --classes "%CLASS_A%" "%CLASS_B%" ^
        --no-profile ^
        %*
)

endlocal
