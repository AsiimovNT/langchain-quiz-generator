@echo off
rem Start the Lang chain Quiz generator and open it in the browser.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run setup.bat first.
    pause
    exit /b 1
)

rem Open the browser a few seconds after the server starts.
start "" /b cmd /c "timeout /t 4 /nobreak >nul && explorer http://localhost:8501"

".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
