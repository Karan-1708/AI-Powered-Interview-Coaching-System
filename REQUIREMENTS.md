### ***Project Requirements Specification***
#### ***Project: AI-Powered Interview Coaching System (MVP Final)***
---

## ***1. Core Functional Requirements***
#### ***Audio Input:***
* Record audio directly from the browser microphone using a native Streamlit component.
* Support for variable recording lengths (optimized for 1–5 minute responses).

#### ***Transcription Engine:***
* **Local Processing:** High-performance local inference via `faster-whisper`.
* **Verbatim Accuracy:** Transcription captures disfluencies ("um", "uh", repetitions) critical for behavioral analysis.

## ***2. Acoustic Analysis Engine ("The Ears")***
#### ***Speaking Rate (WPM):***
* **Logic:** (Total Words / Total Duration) * 60.
* **Context-Aware Targets:** High-accuracy thresholds for Technical (100-120), Standard (130-160), and Presentation (130-150) modes.

#### ***Fluency Metrics:***
* **Pause Detection:** Identification of gaps >1.5 seconds.
* **Filler Counting:** Regex-based detection of um, uh, like, you know, basically, actually.
* **Repetition/Stutter Detection:** Flags hyphenated and immediate word repetitions.

#### ***Tone & Emotion Analysis:***
* **Pitch Analysis:** Pitch variance (F0) tracking via YIN algorithm.
* **Energy Analysis:** RMS volume energy monitoring.
* **Emotional Labeling:** Classifies responses into states like Confident, Nervous, Monotone, or Energetic.

## ***3. Intelligent Evaluation ("The Brain")***
#### ***STAR-Based Feedback:***
* **Dual-Engine Support:** Seamless switching between Local (Ollama) and Frontier (Gemini, OpenAI, Anthropic) LLMs.
* **Structured Evaluation:** Automated analysis of Situation, Task, Action, and Result (STAR) components.
* **Persona-Driven Context:** Context-aware feedback generated based on the specific interviewer persona selected (HR, Tech Lead, or Stress Interviewer).

## ***4. Universal Accessibility ("The Body")***
#### ***Compute Allocation Architecture:***
* **Direct Hardware Selection:** Explicit user control between **NVIDIA GPU** (FP16) or **CPU & RAM Core** (Int8).
* **Self-Healing Inference:** Intelligent Whisper model scaling (tiny/small/medium) based on detected system RAM and VRAM.
* **Cross-Platform:** Native support for Windows, macOS (Apple Silicon), and Linux.

## ***5. Privacy & Data Sovereignty***
* **Local First:** All audio and transcripts stored in `./temp_data`.
* **Zero Data Harvesting:** No data is sent to the cloud unless an External API is intentionally configured by the user.
* **Instant Purge:** "Delete All Data" feature wipes all session history, audio, and diagnostic logs.

## ***6. System Dependencies***
* **Frontend:** streamlit >= 1.30.0
* **Backend:** fastapi, uvicorn
* **AI/ML:** faster-whisper, torch, librosa, numpy, pydantic
* **Utilities:** fpdf2, markdown, plotly, psutil, py-cpuinfo
