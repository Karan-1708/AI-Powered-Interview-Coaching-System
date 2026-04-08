@echo off
TITLE AI Interview Coach
echo ===================================================
echo   AI Interview Coach - Windows Startup
echo ===================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.10, 3.11, 3.12, or 3.13 and ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b
)

:: Setup Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] First time setup: Creating virtual environment...
    python -m venv .venv
    
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
    
    echo [INFO] Upgrading pip...
    python -m pip install --upgrade pip
    
    :: Logic to pick the "Sweet Version" of Torch for Windows
    python -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Python 3.13 detected. Installing Nightly CUDA 12.4 for best compatibility...
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
    ) else (
        echo [INFO] Python 3.10-3.12 detected. Installing Stable CUDA 12.1...
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    )
    
    echo [INFO] Installing remaining dependencies...
    pip install -r requirements.txt
) else (
    echo [INFO] Activating existing virtual environment...
    call .venv\Scripts\activate.bat
)

:: Setup Environment Variables
if not exist ".env" (
    echo [INFO] Creating default .env file from .env.example...
    copy .env.example .env >nul
)

:: Launch Backend
echo.
echo [INFO] Starting FastAPI Backend Server in a new window...
:: Opens a new terminal window for the backend so you can see its logs separately
start "AI Coach - Backend Server" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"

:: Wait for Backend to initialize
echo [INFO] Waiting for backend to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

:: Launch Frontend
echo.
echo [INFO] Starting Streamlit Frontend...
echo [INFO] A browser window should open automatically.
streamlit run app.py

echo.
echo [INFO] Application closed.
pause
