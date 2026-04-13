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

## 🧠 Dual Inference Approach

The system provides maximum flexibility by supporting two distinct inference paths for the AI "Brain":

### 1. Local Inference (Privacy First)
- **Engine**: [Ollama](https://ollama.com/)
- **Target**: Users who require 100% offline, private coaching.
- **Implementation**: The `LLMClient` communicates with a local Ollama instance via its REST API. It handles model probing, tag listing, and conversational chat.

### 2. Cloud Frontier (High Performance)
- **Engine**: REST API (OpenAI, Anthropic, Google Gemini)
- **Target**: Users seeking state-of-the-art reasoning and coaching quality.
- **Implementation**: The `LLMClient` features a unified interface that routes requests to the respective cloud provider. It includes automated parameter handling for newer reasoning models (like OpenAI's O-series).

---

## 🔧 Installation & Setup

### The Automated Way (Recommended)

Our **`install.py`** script is designed to detect your hardware and configure the environment automatically.

1. Open your IDE (VS Code, Cursor, etc.).
2. Open a terminal and run:
  ```bash
    python install.py
  ```
3. **What happens**: The script upgrades pip, uninstalls conflicting CPU versions of PyTorch, detects your OS/Hardware (Windows CUDA vs Apple Silicon vs Linux), installs the "Sweet Version" of the AI engine, downloads portable FFmpeg (on Windows), and sets up your `.env`.

### The Manual Way

If you prefer total control:

1. **Create venv**: `python -m venv .venv`
2. **Activate venv**: `.venv\Scripts\Activate.ps1`
3. **Install Engine**: Use the specific index for your hardware (refer to `install.py` logic for URL links).
4. **Install Requirements**: `pip install -r requirements.txt`
5. **Security**: Copy `.env.example` to `.env` and set your `INTERNAL_API_KEY`.

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