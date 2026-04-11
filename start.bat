@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  AI Interview Coach — Windows Launcher
::  Checks Python, runs setup engine, launches backend+frontend
:: ============================================================

title AI Interview Coach

:: ── ANSI colour support (Windows 10+) ───────────────────────
for /f "tokens=4-5 delims=. " %%i in ('ver') do (
    set WINVER=%%i
)

:: ── Banner ───────────────────────────────────────────────────
echo.
echo  [96m[1m+----------------------------------------------------------+[0m
echo  [96m[1m^|          AI Interview Coach  ^|  Windows Launcher         ^|[0m
echo  [96m[1m+----------------------------------------------------------+[0m
echo.

:: ============================================================
::  PHASE 1 — Ensure Python 3.11+ is available
:: ============================================================
echo  [94m[1m[1/4][0m  Checking Python installation...
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 goto :install_python

:: Python found — check version is 3.11+
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if %PY_MAJOR% LSS 3 goto :python_too_old
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 11 goto :python_too_old

echo  [92m[1m  ✔[0m  Python %PYVER% found — compatible
goto :run_setup

:python_too_old
echo  [93m  ⚠[0m  Python %PYVER% is too old ^(need 3.11+^)
echo  [93m  ⚠[0m  Attempting to install Python 3.12 via winget...
echo.
goto :install_python

:install_python
echo  [96m  ℹ[0m  Installing Python 3.12 (this may take a minute)...
echo.
winget install --id Python.Python.3.12 --exact --silent ^
    --accept-package-agreements --accept-source-agreements

if %errorlevel% neq 0 (
    echo.
    echo  [91m  ✘[0m  Could not install Python automatically.
    echo  [91m      Please download it from: https://www.python.org/downloads/[0m
    echo.
    pause
    exit /b 1
)

echo.
echo  [92m  ✔[0m  Python installed successfully.
echo  [93m  ⚠[0m  Please CLOSE this window and run start.bat again
echo  [93m      so Windows can pick up the new Python installation.[0m
echo.
pause
exit /b 0

:: ============================================================
::  PHASE 2 — Run the Python setup engine (install.py)
:: ============================================================
:run_setup
echo.
echo  [94m[1m[2/4][0m  Running setup engine...
echo.

:: Add local ./bin to PATH so portable FFmpeg is found
set "PATH=%CD%\bin;%PATH%"

python install.py
if %errorlevel% neq 0 (
    echo.
    echo  [91m  ✘[0m  Setup failed. Review the messages above.
    echo  [91m      If the problem persists, delete the .venv folder and try again.[0m
    echo.
    pause
    exit /b 1
)

:: ============================================================
::  PHASE 3 — Activate venv and launch backend
:: ============================================================
echo.
echo  [94m[1m[3/4][0m  Starting backend server...
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo  [91m  ✘[0m  Virtual environment not found after setup. Please try again.
    pause
    exit /b 1
)

:: Add ./bin to PATH inside the venv session too
set "FULL_PATH=%CD%\bin;%CD%\.venv\Scripts;%PATH%"

:: Launch backend in a separate window (stays open if it crashes so user can read errors)
start "AI Interview Coach — Backend" cmd /k ^
    "set PATH=%CD%\bin;%%PATH%% && ^
     call .venv\Scripts\activate.bat && ^
     echo  [96m  ℹ[0m  Backend starting on http://localhost:8000 ... && ^
     python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload"

:: Give the backend a moment to bind its port
echo  [96m  ℹ[0m  Waiting for backend to start...
timeout /t 5 >nul

:: ============================================================
::  PHASE 4 — Launch frontend (Streamlit) in this window
:: ============================================================
echo.
echo  [94m[1m[4/4][0m  Launching dashboard...
echo.
echo  [92m  ✔[0m  Opening AI Interview Coach in your browser...
echo  [96m      (Close this window to shut down the app)[0m
echo.

call .venv\Scripts\activate.bat
set "PATH=%CD%\bin;%PATH%"
streamlit run app.py

:: ── Shutdown message ─────────────────────────────────────────
echo.
echo  [96m  ℹ[0m  Dashboard closed.
echo  [93m  ⚠[0m  The backend server window is still open.
echo  [93m      Close that window too to fully stop the app.[0m
echo.
pause
endlocal