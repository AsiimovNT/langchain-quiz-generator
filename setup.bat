@echo off
rem One-time setup: virtual environment, dependencies and the Mistral GGUF model.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer is required: https://www.python.org/downloads/
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail

rem NVIDIA GPU present -> CUDA build of llama.cpp, otherwise the CPU build.
set REQUIREMENTS=requirements-cpu.txt
where nvidia-smi >nul 2>nul && set REQUIREMENTS=requirements-gpu.txt
echo Installing %REQUIREMENTS% ...
".venv\Scripts\python.exe" -m pip install -r %REQUIREMENTS% || goto :fail

if not exist "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf" (
    echo Downloading the Mistral 7B model, about 4.4 GB...
    ".venv\Scripts\python.exe" scripts\download_model.py || goto :fail
)

echo.
echo Setup complete. Start the app with run.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
