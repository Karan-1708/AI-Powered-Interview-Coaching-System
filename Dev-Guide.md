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

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\Activate.ps1     # Windows

# Install CUDA-enabled Torch (Recommended for NVIDIA GPUs)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
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
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
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