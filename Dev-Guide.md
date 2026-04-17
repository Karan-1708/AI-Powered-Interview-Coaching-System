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
3. **What happens** (7 steps):
   - Checks Python version (3.11–3.13). If none is found, auto-installs Python 3.12 via `winget` (Windows), Homebrew (macOS), or the system package manager (Linux), then relaunches itself.
   - Checks whether Ollama is installed. If not, prompts you and installs it silently if you agree.
   - Upgrades pip and uninstalls any conflicting CPU-only PyTorch builds.
   - Detects OS/hardware (NVIDIA CUDA, Apple Silicon, CPU) and installs the correct PyTorch wheel index (`cu124` for Python ≥ 3.12, `cu121` for Python 3.11).
   - Installs all `requirements.txt` dependencies.
   - Downloads portable **FFmpeg** (Windows only).
   - Creates `.env` from `.env.example` if it doesn't exist and generates a random `INTERNAL_API_KEY`.

### The Manual Way

If you prefer total control:

1. **Create venv**: `python -m venv ai-venv`
2. **Activate venv**: `ai-venv\Scripts\Activate.ps1` (Windows) or `source ai-venv/bin/activate` (Mac/Linux)
3. **Install Engine**: Use the specific index for your hardware (refer to `install.py` logic for URL links).
4. **Install Requirements**: `pip install -r requirements.txt`
5. **Security**: Copy `.env.example` to `.env` and set your `INTERNAL_API_KEY`.
6. **Optional — Pre-load API Keys**: Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` in `.env` and the app will load them on startup — no need to enter them in the UI every session.

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

## 🎭 Interviewer Personas

Personas are defined in `src/backend/personas.py`. There are six:

| Label | Routing Keywords |
|---|---|
| 🤝 Friendly HR | `hr`, `screen`, `phone`, `recruiter`, `initial` |
| 🔬 Strict Tech Lead | `technical`, `system`, `code`, `design`, `architecture` |
| 🎯 Behavioral Coach | `behavioral`, `behaviour`, `star`, `competency`, `situational` |
| 🌱 Culture Fit | `culture`, `values`, `fit`, `team`, `peer`, `onsite` |
| 🔥 Stress Interviewer | (default fallback) |
| 🏛️ Executive Sponsor | `executive`, `vp`, `director`, `c-level`, `strategic`, `final` + senior seniority |

Routing is handled by `Personas.get_interviewer_by_type(round_type, seniority)`. The resulting system prompt is further modulated by one of four **seniority modifiers** (Entry-Level → Executive) injected at the end of the base prompt via `get_interview_sys_prompt()`.

To add a new persona: define a new dict with `label`, `icon`, `base_prompt`, and `recommended_mode` keys, then register it in `PERSONA_PROMPTS` and add its routing keywords to `get_interviewer_by_type`.

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

The system is fully containerized with two services: `api` (FastAPI) and `ui` (Streamlit).

```bash
docker compose up --build
```

**Key details:**
- **Base image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Ubuntu 22.04 — required for clean ffmpeg dependencies).
- **Healthcheck**: The API service polls `GET /health` every 15 s with a 60 s `start_period` to allow faster-whisper models to load. The UI container will not start until the API passes its healthcheck (`depends_on: condition: service_healthy`).
- **Secrets**: Neither container bakes in an `INTERNAL_API_KEY`. Drop a `.env` file next to `docker-compose.yml` before starting — it is loaded by both services via `env_file: required: false`.
- **Volumes**: `./temp_data` and `./logs` are mounted into both containers so recordings and logs persist on the host.
- **NVIDIA GPU passthrough**: The API service requests all available NVIDIA devices automatically via the `deploy.resources` block. Non-GPU hosts simply ignore this.

```bash
# Bring down and wipe volumes
docker compose down -v
```