@echo off
setlocal enabledelayedexpansion
TITLE AI Interview Coach

echo.
echo  +----------------------------------------------------------+
echo  ^|          AI Interview Coach  ^|  Windows Launcher        ^|
echo  +----------------------------------------------------------+
echo.

REM ── Add local ./bin to PATH for portable FFmpeg ──────────────
set "PATH=%~dp0bin;%PATH%"

set TOTAL=5

REM ============================================================
REM  PHASE 1 — Locate Python 3.11+
REM ============================================================
echo [1/%TOTAL%] Checking Python Installation...
echo.

set MIN_MINOR=11
set PYTHON_CMD=

REM Try the Windows py launcher first — most reliable on Windows
where py >nul 2>&1
if %errorlevel% equ 0 (
    for %%v in (3.13 3.12 3.11) do (
        py -%%v --version >nul 2>&1
        if !errorlevel! equ 0 (
            for /f "delims=" %%p in ('py -%%v -c "import sys; print(sys.executable)" 2^>nul') do (
                set "PYTHON_CMD=%%p"
            )
            goto :python_found
        )
    )
)

REM Try common executable names
for %%c in (python3.12 python3.11 python3 python) do (
    where %%c >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%v in ('%%c --version 2^>^&1') do (
            for /f "tokens=1,2 delims=." %%a in ("%%v") do (
                if %%a geq 3 (
                    if %%b geq %MIN_MINOR% (
                        set "PYTHON_CMD=%%c"
                        goto :python_found
                    )
                )
            )
        )
    )
)

REM ── Python 3.11+ not found — try to install ──────────────────
echo   [WARN] Python 3.11+ not found on this system.
echo   [INFO] Attempting to install Python 3.12 via winget...
echo.

winget install --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
if %errorlevel% equ 0 (
    echo   [OK] Python 3.12 installed successfully.
    echo   [INFO] Please close this window and re-run start.bat so the new Python is detected.
    pause
    exit /b 0
)

echo   [WARN] winget install failed.
echo   [INFO] Please install Python 3.11+ manually:
echo          https://www.python.org/downloads/
echo          Make sure to tick "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:python_found
for /f "tokens=2" %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do (
    echo   [OK] Python %%v found
)

REM ============================================================
REM  PHASE 2 — Check / install Ollama
REM ============================================================
echo.
echo [2/%TOTAL%] Checking Ollama (Local AI Engine)...
echo.

where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Ollama is already installed
    ollama list >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [INFO] Ollama service is running
    ) else (
        echo   [INFO] Ollama installed -- run "ollama serve" to start the local model server
    )
    goto :ollama_done
)

echo   [WARN] Ollama not found on this system.
echo   [INFO] Ollama enables free local AI inference -- no API key needed.
echo   [INFO] You can skip this and use OpenAI / Gemini / Anthropic instead.
echo.
set /p OLLAMA_CHOICE="  Install Ollama now? [Y/n]: "
if /i "!OLLAMA_CHOICE!"=="n"  goto :skip_ollama
if /i "!OLLAMA_CHOICE!"=="no" goto :skip_ollama

echo   [INFO] Downloading Ollama installer...
set "OLLAMA_TMP=%TEMP%\OllamaSetup.exe"
curl -fsSL "https://ollama.com/download/OllamaSetup.exe" -o "%OLLAMA_TMP%"
if %errorlevel% neq 0 (
    echo   [WARN] Download failed. Install manually from https://ollama.com
    goto :ollama_done
)
echo   [INFO] Running Ollama installer...
"%OLLAMA_TMP%" /SILENT
if %errorlevel% equ 0 (
    echo   [OK] Ollama installed successfully
    del /f /q "%OLLAMA_TMP%" >nul 2>&1
) else (
    echo   [WARN] Installer failed. Install manually from https://ollama.com
    del /f /q "%OLLAMA_TMP%" >nul 2>&1
)
goto :ollama_done

:skip_ollama
echo   [INFO] Skipping Ollama. Install later from https://ollama.com

:ollama_done

REM ============================================================
REM  PHASE 3 — Run the Python setup engine (install.py)
REM ============================================================
echo.
echo [3/%TOTAL%] Running Setup Engine...
echo.

if not exist "install.py" (
    echo   [ERROR] install.py not found.
    echo   [INFO]  Make sure you are running this script from the project root folder.
    pause
    exit /b 1
)

"%PYTHON_CMD%" install.py
if %errorlevel% neq 0 (
    echo.
    echo   [ERROR] Setup failed. Review the messages above.
    echo   [INFO]  If the problem persists, delete the .venv folder and try again.
    pause
    exit /b 1
)

REM ── Activate the venv that install.py created ────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo   [ERROR] Virtual environment was not created. Please re-run this script.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

REM Keep ./bin on PATH inside the venv session
set "PATH=%~dp0bin;%PATH%"

REM ============================================================
REM  PHASE 4 — Start backend server (new window)
REM ============================================================
echo.
echo [4/%TOTAL%] Starting Backend Server...
echo.

echo   [INFO] Launching FastAPI backend on http://localhost:8000 ...
start "AI Coach Backend" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"

REM Poll until the backend responds (up to 30 s)
echo   [INFO] Waiting for backend to be ready...
set READY=0
for /l %%i in (1,1,30) do (
    if !READY! equ 0 (
        curl -s --max-time 1 http://localhost:8000/health >nul 2>&1
        if !errorlevel! equ 0 (
            set READY=1
        ) else (
            timeout /t 1 /nobreak >nul
        )
    )
)

if %READY% equ 1 (
    echo   [OK] Backend is ready
) else (
    echo   [WARN] Backend health check timed out -- it may still be loading.
    echo   [WARN] If the app does not work, check if port 8000 is already in use.
)

REM ============================================================
REM  PHASE 5 — Launch Streamlit frontend (foreground)
REM ============================================================
echo.
echo [5/%TOTAL%] Launching Dashboard...
echo.
echo   [OK] Opening AI Interview Coach in your browser...
echo   [INFO] Press Ctrl+C in this window to stop the app.
echo.

streamlit run app.py

echo.
echo   [INFO] Application closed.
pause
