#!/usr/bin/env bash

# ============================================================
#  AI Interview Coach — Mac / Linux Launcher
#  Checks Python, runs setup engine, launches backend+frontend
# ============================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
RED="\033[91m"
GREEN="\033[92m"
YELLOW="\033[93m"
BLUE="\033[94m"
CYAN="\033[96m"
WHITE="\033[97m"

ok()   { echo -e "  ${GREEN}✔${RESET}  $*"; }
info() { echo -e "  ${CYAN}ℹ${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
err()  { echo -e "  ${RED}✘${RESET}  ${RED}$*${RESET}"; }

fatal() {
    err "$*"
    echo -e "\n${RED}${BOLD}  Setup cannot continue. Fix the issue above and try again.${RESET}\n"
    exit 1
}

section() {
    local n=$1 total=$2 title=$3
    echo -e "\n${BOLD}${BLUE}[${n}/${total}]${RESET}  ${WHITE}${title}${RESET}"
}

# ── Banner ───────────────────────────────────────────────────
echo -e """
${CYAN}${BOLD}+----------------------------------------------------------+
|          AI Interview Coach  |  Mac / Linux Launcher     |
+----------------------------------------------------------+${RESET}
"""

# Add local ./bin to PATH now so portable FFmpeg is available throughout
export PATH="$(pwd)/bin:$PATH"

TOTAL=4

# ============================================================
#  PHASE 1 — Locate / install Python 3.11+
# ============================================================
section 1 $TOTAL "Checking Python Installation"

MIN_MAJOR=3
MIN_MINOR=11

_find_python() {
    # Try python3 first, then python
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" --version 2>&1 | awk '{print $2}')
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -gt "$MIN_MAJOR" ] || \
               ([ "$major" -eq "$MIN_MAJOR" ] && [ "$minor" -ge "$MIN_MINOR" ]); then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=""
if ! PYTHON_CMD=$(_find_python 2>/dev/null); then
    warn "Python ${MIN_MAJOR}.${MIN_MINOR}+ not found — attempting to install..."

    OS_TYPE="$(uname -s)"
    if [ "$OS_TYPE" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            info "Installing Python 3.12 via Homebrew..."
            brew install python@3.12
            export PATH="$(brew --prefix python@3.12)/bin:$PATH"
        else
            fatal "Homebrew not found.\nInstall it from https://brew.sh then re-run this script.\nOr install Python manually from https://www.python.org/downloads/"
        fi
    elif [ "$OS_TYPE" = "Linux" ]; then
        if command -v apt-get &>/dev/null; then
            info "Installing Python 3.12 via apt..."
            sudo apt-get update -qq
            sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
        elif command -v dnf &>/dev/null; then
            info "Installing Python 3.12 via dnf..."
            sudo dnf install -y python3.12
        else
            fatal "Could not auto-install Python.\nPlease install Python ${MIN_MAJOR}.${MIN_MINOR}+ from https://www.python.org/downloads/"
        fi
    fi

    if ! PYTHON_CMD=$(_find_python 2>/dev/null); then
        fatal "Python ${MIN_MAJOR}.${MIN_MINOR}+ still not found after install attempt.\nPlease install it manually and re-run this script."
    fi
fi

PY_VER=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
ok "Python $PY_VER found at $(command -v $PYTHON_CMD)"

# ============================================================
#  PHASE 2 — Run the Python setup engine (install.py)
# ============================================================
section 2 $TOTAL "Running Setup Engine"

if [ ! -f "install.py" ]; then
    fatal "install.py not found. Make sure you are running this script from the project root folder."
fi

"$PYTHON_CMD" install.py
if [ $? -ne 0 ]; then
    echo ""
    err "Setup failed. Review the messages above."
    warn "If the problem persists, delete the .venv folder and try again:"
    warn "  rm -rf .venv && bash start.sh"
    exit 1
fi

# ── Activate the venv that install.py created ────────────────
if [ ! -f ".venv/bin/activate" ]; then
    fatal "Virtual environment was not created. Please re-run this script."
fi

source .venv/bin/activate

# Keep ./bin on PATH inside the venv session
export PATH="$(pwd)/bin:$PATH"

# ============================================================
#  PHASE 3 — Start backend server
# ============================================================
section 3 $TOTAL "Starting Backend Server"

info "Launching FastAPI backend on http://localhost:8000 ..."

# Launch in background; capture PID so we can kill it on exit
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap Ctrl+C and normal exit to cleanly kill the backend
_cleanup() {
    echo -e "\n\n${CYAN}  ℹ${RESET}  Shutting down backend server (PID $BACKEND_PID)..."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
    echo -e "${GREEN}  ✔${RESET}  All services stopped. Goodbye!"
    exit 0
}
trap _cleanup INT TERM EXIT

# Wait for backend to be ready (poll up to 15 s)
info "Waiting for backend to be ready..."
READY=0
for i in $(seq 1 15); do
    if curl -s --max-time 1 http://localhost:8000/health >/dev/null 2>&1 ||
       curl -s --max-time 1 http://localhost:8000/      >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -eq 1 ]; then
    ok "Backend is ready"
else
    warn "Backend health check timed out — it may still be loading."
    warn "If the app doesn't work, check if port 8000 is already in use."
fi

# ============================================================
#  PHASE 4 — Launch Streamlit frontend (foreground)
# ============================================================
section 4 $TOTAL "Launching Dashboard"

echo ""
ok "Opening AI Interview Coach in your browser..."
info "(Press Ctrl+C in this terminal to stop the app)"
echo ""

streamlit run app.py

# _cleanup runs automatically on exit via trap