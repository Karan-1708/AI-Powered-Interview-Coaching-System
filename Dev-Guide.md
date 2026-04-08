# 🛠️ Developer Guide: AI Interview Coach

This guide is for developers who want to modify the source code, run tests, or integrate this system into a larger architecture.

---

## 🏗️ Architecture Overview

The system is built as a decoupled **Client-Server** application:

- **Backend (FastAPI)**: Handles heavy lifting: Whisper transcription, LLM API routing, local Ollama orchestration, and hardware telemetry.
- **Frontend (Streamlit)**: A reactive dashboard for audio recording, chat UI, and Plotly analytics.
- **Acoustic Engine**: Uses `librosa` and `faster-whisper` for real-time speech-to-text and metric extraction.

---

## 🔧 Manual Installation

### 1. Environment Setup

We recommend using **Python 3.10 to 3.13**.

#### The Quick Way (Recommended)

Open your IDE (VS Code, Cursor, etc.) and run the following in your terminal:

```bash
python -m venv .venv
# Activate your venv, then:
python install.py
```

This script will automatically detect your OS and hardware to install the optimized version of PyTorch and all dependencies.

#### The Manual Way

If you prefer total control:

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.\venv\Scripts\activate     # Windows

```

### 2. Dependency Management

Install all core libraries:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)

Create a `.env` file in the root directory. Copy the structure from `.env.example`:

- `INTERNAL_API_KEY`: Used for authentication between Frontend and Backend.
- `API_URL`: The URL where the Backend is reachable (default `http://127.0.0.1:8000`).
- `OLLAMA_HOST`: The local endpoint for Ollama (default `http://127.0.0.1:11434`).

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

*Note: Ensure you have the NVIDIA Container Toolkit installed if you want GPU support inside Docker.*