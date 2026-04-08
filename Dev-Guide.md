Data Drifters Logo

# 🛠️ Developer Guide: AI Interview Coach

This guide is for developers who want to modify the source code, run tests, or integrate this system into a larger architecture.

---

## 🏗️ Architecture Overview

The system is built as a decoupled **Client-Server** application:

- **Backend (FastAPI)**: Manages AI heavy-lifting (Whisper, LLM routing, and acoustic scoring).
- **Frontend (Streamlit)**: A reactive dashboard for audio recording, chat UI, and Plotly analytics.
- **Acoustic Engine**: Uses `librosa` and `faster-whisper` for real-time speech-to-text and metric extraction.

---

## 🔧 Installation & Setup

### The Automated Way (Recommended)

Our `**install.py`** script is designed to detect your hardware and configure the environment automatically.

1. Open your IDE (VS Code, Cursor, etc.).
2. Open a terminal and run:
  ```bash
    python install.py
  ```
3. **What happens**: The script upgrades pip, uninstalls conflicting CPU versions of PyTorch, detects your OS/Hardware (Windows CUDA vs Apple Silicon vs Linux), installs the "Sweet Version" of the AI engine, downloads portable FFmpeg (on Windows), and sets up your `.env`.

### The Manual Way

If you prefer total control:

1. **Create venv**: `python -m venv .venv`
2. **Install Engine**: Use the specific index for your hardware (refer to `install.py` logic for URL links).
3. **Install Requirements**: `pip install -r requirements.txt`
4. **Security**: Copy `.env.example` to `.env` and set your `INTERNAL_API_KEY`.

---

## 🚦 Execution

### Start Backend

```bash
uvicorn src.api.server:app --reload
```

### Start Frontend

```bash
streamlit run app.py
```

---

## 🧪 Testing & Diagnostics

We use `pytest` for the API and utility suites:

```bash
# Run all tests
$env:PYTHONPATH = "."; pytest tests/
```

### Resource Monitoring

The backend logs detailed hardware info on startup. Look for:

- `INFO: CUDA: Detected [OK]` -> Confirms GPU acceleration is active.
- `INFO: FFmpeg: Detected [OK]` -> Confirms audio parsing is ready.

---

## 🐳 Docker Deployment

The system is fully containerized.

```bash
docker-compose up --build
```

*Note: The containers automatically handle dependency installation for standard Linux/NVIDIA environments.*