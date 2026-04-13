@echo off
TITLE AI Interview Coach
echo ===================================================
echo   AI Interview Coach - Startup
echo ===================================================

:: 1. Run Setup/Install logic
python install.py
if %errorlevel% neq 0 (
    echo [ERROR] Setup failed. Please check the logs above.
    pause
    exit /b
)

:: 2. Launch Backend
echo.
echo [3/4] Starting backend server...
:: Opens a new terminal window for the backend
start "AI Coach Backend" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"

:: 3. Wait for the Backend
echo [INFO] Waiting for backend to start...
timeout /t 5 /nobreak >nul

:: 4. Launch the Frontend
echo.
echo [4/4] Launching dashboard...
echo [INFO] Opening AI Interview Coach in your browser...
call .venv\Scripts\activate.bat && streamlit run app.py

echo.
echo [INFO] Application closed.
pause
