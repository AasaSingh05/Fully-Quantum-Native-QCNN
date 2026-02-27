@echo off
setlocal enabledelayedexpansion

:: Windows Launcher for QCNN Training
:: Usage: runApp.bat [type] [path] [encoding] [image_size] [class1 class2]

:: Handle flags passed as first argument
set "first_arg=%~1"
if "!first_arg:~0,2!"=="--" (
    echo Detected flags. Using default Research Configuration (MNIST-IDX/Amplitude)...
    python main.py --dataset idx --path datasets/MNIST/ --samples 500 --encoding amplitude --learning-rate 0.005 --no-profile %*
    exit /b %errorlevel%
)

:: Configuration with defaults
set "DATASET_TYPE=%~1"
if "%DATASET_TYPE%"=="" set "DATASET_TYPE=idx"

set "DATASET_PATH=%~2"
if "%DATASET_PATH%"=="" set "DATASET_PATH=datasets/MNIST/"

set "ENCODING=%~3"
if "%ENCODING%"=="" set "ENCODING=amplitude"

set "IMAGE_SIZE=%~4"
if "%IMAGE_SIZE%"=="" set "IMAGE_SIZE=28"

set "SAMPLES=%~5"
if "%SAMPLES%"=="" set "SAMPLES=500"

set "LEARNING_RATE=%~6"
if "%LEARNING_RATE%"=="" set "LEARNING_RATE=0.005"

:: Shift arguments
shift
shift
shift
shift
shift
shift

echo ------------------------------------------------
echo Quantum Native QCNN - Research Accuracy Launcher (Win)
echo ------------------------------------------------
echo Dataset: %DATASET_TYPE%
echo Encoding: %ENCODING% (Size: %IMAGE_SIZE%)
echo Samples: %SAMPLES% | LR: %LEARNING_RATE%
echo ------------------------------------------------

:: Check for virtual environment
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py ^
        --dataset "%DATASET_TYPE%" ^
        --path "%DATASET_PATH%" ^
        --encoding "%ENCODING%" ^
        --image-size "%IMAGE_SIZE%" ^
        --samples "%SAMPLES%" ^
        --learning-rate "%LEARNING_RATE%" ^
        --no-profile ^
        %*
) else (
    python main.py ^
        --dataset "%DATASET_TYPE%" ^
        --path "%DATASET_PATH%" ^
        --encoding "%ENCODING%" ^
        --image-size "%IMAGE_SIZE%" ^
        --samples "%SAMPLES%" ^
        --learning-rate "%LEARNING_RATE%" ^
        --no-profile ^
        %*
)

endlocal
