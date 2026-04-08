#!/bin/bash

echo "==================================================="
echo "  AI Interview Coach - Mac/Linux Startup"
echo "==================================================="

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 is not installed or not in your system PATH."
    echo "Please install Python 3.10-3.13."
    exit 1
fi

# Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[INFO] First time setup: Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    
    echo "[INFO] Activating virtual environment..."
    source .venv/bin/activate
    
    echo [INFO] Upgrading pip...
    pip install --upgrade pip

    echo [INFO] Installing optimized AI Engine (Torch)...
    if [ "$(uname)" == "Darwin" ]; then
        # MacOS - Check for Apple Silicon (M-series) vs Intel
        if [ "$(uname -m)" == "arm64" ]; then
            echo "[INFO] Apple Silicon detected. Installing Torch with MPS/Metal support..."
            pip install torch
        else
            echo "[INFO] Apple Intel detected. Installing standard Torch..."
            pip install torch
        fi
    else
        # Linux - Check Python version for CUDA wheels
        PYTHON_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$PYTHON_VER" == "3.13" ]]; then
            echo "[INFO] Linux + Python 3.13 detected. Installing Nightly CUDA 12.4..."
            pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
        else
            echo "[INFO] Linux detected. Installing Stable CUDA 12.1..."
            pip install torch --index-url https://download.pytorch.org/whl/cu121
        fi
    fi

    echo [INFO] Installing remaining dependencies...
    pip install -r requirements.txt

else
    echo "[INFO] Activating existing virtual environment..."
    source .venv/bin/activate
fi

# Setup Environment Variables
if [ ! -f ".env" ]; then
    echo "[INFO] Creating default .env file from .env.example..."
    cp .env.example .env
fi

echo ""
echo "[INFO] Starting FastAPI Backend Server..."
# Start backend in the background
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to initialize
echo "[INFO] Waiting for backend to initialize (5 seconds)..."
sleep 5

echo ""
echo "[INFO] Starting Streamlit Frontend..."
echo "[INFO] A browser window should open automatically."
echo "[INFO] Press Ctrl+C to stop both Frontend and Backend."

# Trap Ctrl+C to kill the background backend process when stopping the frontend
trap "kill $BACKEND_PID; exit" INT TERM EXIT

# Start frontend in the foreground
streamlit run app.py
