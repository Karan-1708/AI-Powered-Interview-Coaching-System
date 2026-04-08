@echo off
TITLE AI Interview Coach - Setup
echo ===================================================
echo   AI Interview Coach - Ultimate Setup
echo ===================================================

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python not found. Attempting to install via Winget...
    winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [ERROR] Automatic Python installation failed.
        echo Please download Python 3.12 manually from https://python.org
        pause
        exit /b
    )
    echo [SUCCESS] Python installed. Please RESTART this script to continue.
    pause
    exit /b
)

:: 2. Check for FFmpeg (Local Portable Version)
if not exist "bin\ffmpeg.exe" (
    echo [INFO] FFmpeg not found. Downloading portable version...
    mkdir bin >nul 2>&1
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'ffmpeg.zip'"
    echo [INFO] Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'bin_temp'"
    :: Move the actual exe to our bin folder
    powershell -Command "Get-ChildItem -Path 'bin_temp' -Filter 'ffmpeg.exe' -Recurse | Move-Item -Destination 'bin\'"
    powershell -Command "Get-ChildItem -Path 'bin_temp' -Filter 'ffprobe.exe' -Recurse | Move-Item -Destination 'bin\'"
    :: Cleanup
    del ffmpeg.zip
    rmdir /s /q bin_temp
    echo [SUCCESS] Portable FFmpeg ready!
)

:: Add local bin to current session path
set PATH=%CD%\bin;%PATH%

:: 3. Setup Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Upgrading pip...
    python -m pip install --upgrade pip
    
    :: Pick Sweet Version
    python -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Installing Nightly CUDA 12.4...
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall --no-cache-dir
    ) else (
        echo [INFO] Installing Stable CUDA 12.1...
        pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir
    )
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: 4. Setup .env
if not exist ".env" copy .env.example .env >nul

:: 5. Launch
echo [INFO] Starting Backend...
start "AI Coach Backend" cmd /k "set PATH=%CD%\bin;%%PATH%% && call .venv\Scripts\activate.bat && python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"

echo [INFO] Starting Frontend...
timeout /t 5 >nul
streamlit run app.py
