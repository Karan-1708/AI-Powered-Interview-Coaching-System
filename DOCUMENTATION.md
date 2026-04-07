# 🛠️ Technical Documentation: AI Interview Coach

This document provides a deep dive into the system architecture, module responsibilities, and data flow for developers looking to enhance or maintain the project.

---

## 🏗️ 1. Architecture Overview
The application follows a **Decoupled Client-Server** pattern:
*   **Frontend (Streamlit)**: Handles user interaction, audio recording, real-time visualization, and report rendering.
*   **Backend (FastAPI)**: Manages heavy AI workloads, hardware telemetry, LLM orchestration, and session state.
*   **Security Layer**: Inter-service communication is secured via a mandatory `X-Internal-Key` header, verified against the environment variable `INTERNAL_API_KEY`.

---

## 📡 2. Backend API (FastAPI)
**Location**: `src/api/server.py`

### Key Endpoints:
*   `POST /process-audio`: Receives a `.wav` file. Triggers the `AudioProcessor` for transcription and the `AcousticScorer` for metrics. Returns JSON metadata.
*   `POST /generate-response`: Standard LLM interface. Injects Resume and Job Description context dynamically into the system prompt.
*   `POST /generate-questions`: Specialized logic for Step 3. Uses role metadata to generate 3 targeted interview questions.
*   `POST /generate-speech`: TTS endpoint using `edge-tts`. Returns a `FileResponse` (MP3).
*   `GET /hardware`: Returns live CPU/RAM/VRAM stats using the `ResourceMonitor`.

### Security Implementation:
The `verify_internal_key` dependency is injected into every sensitive route. If `INTERNAL_API_KEY` is not set in `.env`, the server enters a "Fail-Safe" mode and refuses all requests with a `500` error to prevent unauthorized access.

---

## 👂 3. AI Processing Engines

### 3.1 Transcription (`AudioProcessor`)
**Location**: `src/backend/audio_processor.py`
*   **Engine**: `faster-whisper` (CTranslate2 backend).
*   **Hardware Awareness**: Dynamically selects model size based on available RAM/VRAM:
    *   **NVIDIA GPU**: Loads `medium.en` in `float16`.
    *   **Apple Silicon**: Loads `medium.en` in `float32` (optimized for AMX).
    *   **CPU**: Loads `small.en` or `tiny.en` in `int8` quantization.
*   **Optimization**: Implements a `_model_cache` singleton to prevent reloading the model on every request.

### 3.2 Acoustic Metrics (`AcousticScorer`)
**Location**: `src/backend/scorer.py`
*   **Signal Processing**: Uses `librosa` for volume (RMS) and pitch (YIN algorithm) analysis.
*   **Fluency Detection**: Uses Regex patterns (`FILLER_PATTERN`, `STUTTER_PATTERN`) to count "um", "uh", and repetitions.
*   **Tone Classification**: Heuristic logic that maps Pitch Variation and Energy into emotional labels (e.g., "Confident", "Nervous").

### 3.3 LLM Orchestration (`LLMClient`)
**Location**: `src/backend/llm_client.py`
*   **Providers**: Unified interface for Ollama, OpenAI, Anthropic, and Google Gemini.
*   **Reasoning Model Support**: Automatically detects `o1`, `o3`, `o4`, or `gpt-5` series models and swaps `max_tokens` for `max_completion_tokens`.
*   **System Prompt Injection**: For models that don't support the `system` role (like O-series), it intelligently wraps instructions into the first user message.

---

## 📊 4. Frontend Logic (Streamlit)
**Location**: `app.py` & `src/ui/`

### Key Design Patterns:
*   **Fragments (`@st.fragment`)**: 
    *   `unified_status_monitor`: Isolated heartbeat that refreshes every 5s without greying out the main UI.
    *   `isolated_recorder_flow`: Prevents the audio widget from resetting when background telemetry runs.
*   **Session State**: Tracks the complex `setup_step` (1-3) and maintains the `chat_history` and `aggregated_metrics` for final analysis.
*   **Audio Autoplay**: Injects hidden HTML/JS `<audio>` components with a `nonce` to bypass browser caching and force autoplay of coach responses.

---

## 📂 5. Utilities & Data Persistence

*   **`src/utils/history.py`**: Manages `temp_data/session_history.json`. Implements gamification logic (calculating average WPM and filler trends).
*   **`src/utils/pdf_generator.py`**: Uses `fpdf2` to transform the session transcript and Plotly charts into a professional PDF report.
*   **`src/utils/text_processor.py`**: Robust cleaning for LLM outputs (stripping brackets, JSON formatting, and numbering).
*   **`src/utils/diagnostics.py`**: Global error handling, logging, and Windows DLL mapping for NVIDIA support.

---

## 🛠️ 6. Development Workflow

### Adding a New Persona
1.  Open `src/backend/personas.py`.
2.  Add a new configuration to the `PERSONA_PROMPTS` dictionary.
3.  The frontend will automatically pick up the new style in the Step 3 dropdown.

### Modifying the API Key
Change `INTERNAL_API_KEY` in your `.env` file. The `start.bat` script handles this synchronization for local users, but for cloud users, this must be set in the Streamlit Cloud Secrets dashboard.

### Debugging
All backend errors are logged to `logs/app_debug.log` with a full traceback. The `safe_execute` decorator ensures the server doesn't crash during individual failures.

---

## 🧪 7. Testing Suite
**Location**: `tests/`

The project uses `pytest` for validation. The suite is divided into three main areas:

### 7.1 API Integration Tests (`test_api.py`)
*   **Purpose**: Validates all FastAPI endpoints using `TestClient`.
*   **Key Tests**:
    *   **Health Check**: Ensures the `/` root returns the online status.
    *   **Mocked AI**: Uses `unittest.mock` to simulate LLM and Whisper processing, ensuring the API logic handles data correctly without requiring a GPU during testing.
    *   **Security Check**: Verifies that the `X-Internal-Key` header correctly blocks unauthorized requests.

### 7.2 Utility Tests (`test_utils.py`)
*   **Purpose**: Validates file management and session persistence.
*   **Key Tests**:
    *   **History Persistence**: Checks that session metrics (WPM, fillers) are saved and loaded correctly from `session_history.json`.
    *   **Cleanup Logic**: Ensures the "Delete All Data" function wipes temp files while protecting active system logs.

### 7.3 Concurrency & Stress Testing (`stress_test.py`)
*   **Purpose**: Simulates high-load scenarios (e.g., multiple users transcribing simultaneously).
*   **Usage**: Run manually via `python tests/stress_test.py` to check for race conditions in the `AudioProcessor` cache.
