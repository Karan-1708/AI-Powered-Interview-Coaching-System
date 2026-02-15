### ***Project Requirements Specification***
#### ***Project: AI-Powered Interview Coaching System (Capstone)***
---

## ***1. Core Functional Requirements***
#### ***Audio Input:***

Record audio directly from the browser microphone using a native Streamlit component (Zero-dependency).
Support for variable recording lengths (1 minute to 5 minutes).

#### ***Transcription Engine:***

* **Local Processing:**
All transcription must happen locally on the device (Privacy-First).

* **Engine:**
faster-whisper (CTranslate2 backend) for high-performance inference.

* **Verbatim Accuracy:**
Transcription must capture disfluencies ("um", "uh", stutters) for analysis.

## ***2. Acoustic Analysis Engine (The "Ears")***
***Speaking Rate (WPM):***

**Logic:** Total Words / Total Duration (Industry Standard).

**Context-Aware Targets:**

* *Technical/Complex:* 100–120 WPM

* *Standard Interview:* 130–160 WPM

* *Presentation:* 130–150 WPM

***Fluency Metrics:***

* **Pause Detection:** Flag silences >1.5 seconds.

* **Filler Words:** Count common fillers (um, uh, like, you know).

* **Stutter Detection:** Identify hyphenated repetitions (e.g., "I-I", "the-the").

* **Blunder Detection:** Flag broken sentences trailing off with "...".

***Tone & Emotion Analysis:***

* Analyze Pitch Variance (F0) and RMS Energy (Volume).

* Classify speech into emotional states:

* *Nervous:* (Fast Pace + Low Volume + High Pitch Var)

* *Monotone:* (Low Pitch Var + Moderate Volume)

* *Confident:* (Steady Pace + Moderate Pitch Var)

## ***3. Universal Accessibility (The "Body")***
***3-Tier Performance Architecture:***

* 🟢 **Eco Mode:** tiny.en (Int8) for Low-Spec CPUs (<4GB RAM).

* 🟡 **Balanced Mode:** small.en for Mid-Range CPUs (Apple Silicon / Intel i5).

* 🔴 **Pro Mode:** medium.en (Float16) for High-End NVIDIA GPUs (>4GB VRAM).

***Hardware Agnostic:***

* Auto-detection of OS (Windows/macOS/Linux).

* Auto-switching between CUDA (NVIDIA), CPU (Intel/AMD), and ARM (Apple).

***Self-Healing:***

* Automatic fallback to Eco Mode if VRAM is exceeded (OOM Protection).

## ***4. Intelligent Evaluation (The "Brain" - Upcoming Phase 4)***
***STAR-Based Evaluation:***

* Use Local LLM (Llama 3 / Mistral via Ollama).

* Detect presence of Situation, Task, Action, Result.

***Relevance Check:***

* Semantic comparison between the Interview Question and the User Answer.

***Latency Requirement:***

* Feedback must be generated and displayed within ≤20 seconds of recording completion.

## ***5. Privacy & Data Sovereignty***
* **Local Storage:** No data sent to the cloud. All temporary files stored in ./temp_data.

* **User Control:** "Delete All Data" button to instantly wipe recordings and logs.

* **Transparency:** Real-time counter of stored artifacts in the UI sidebar.

## ***6. System Dependencies***
**Python:** 3.10+

**UI:** streamlit

**AI/ML:** faster-whisper, torch, librosa, numpy

**Hardware Monitoring:** psutil, py-cpuinfo

**LLM Interface:** ollama (Python Client)