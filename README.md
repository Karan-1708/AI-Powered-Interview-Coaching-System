# 🎙️ AI-Powered Interview Coaching System (MVP)

### *Overview*
The **AI-Powered Interview Coaching System** is a locally hosted, privacy-first application designed to democratize access to high-quality interview preparation. It provides a secure, offline environment for candidates to practice behavioral and technical interviews, leveraging advanced AI for real-time transcription, acoustic analysis, and semantic feedback.

---

## 🚀 Key Features

### 🧠 Dual-Engine LLM Support
- **Local Inference:** Seamless integration with **Ollama** (Llama 3, Mistral, Gemma) for 100% offline processing.
- **Cloud Frontier:** Support for **OpenAI (GPT-4o)**, **Anthropic (Claude 3.5)**, and **Google Gemini** for high-performance coaching when internet is available.
- **Smart Connection Testing:** Built-in diagnostic tools to verify API and Local LLM status.

### 👂 Advanced Acoustic Analysis ("The Ears")
- **Real-time Transcription:** Powered by `faster-whisper` for high-speed, verbatim accuracy.
- **Pacing & Fluency:** Automatically calculates **Words Per Minute (WPM)** and flags "um", "uh", and "like" filler words.
- **Explicit Compute Allocation:** Choose between **NVIDIA GPU** or **CPU & RAM Core** directly in the UI to optimize performance for your specific hardware.
- **Hardware-Agnostic Engine:** Intelligent backend that auto-detects CUDA, CoreML, or standard CPU instructions.

### 🎭 Multi-Persona Interview Simulation
- **Dynamic Interviewers:** Choose between a **Friendly HR Recruiter**, a **Strict Technical Lead**, or a high-pressure **Stress Interviewer**.
- **Context-Aware Questions:** Generates role-specific questions based on your industry, job title, and seniority.
- **STAR Framework Evaluation:** AI Coach provides structured feedback based on the Situation, Task, Action, and Result (STAR) method.

### 📊 Dashboard & Reporting
- **Interactive Analytics:** Visualize your pacing and filler word trends over time with Plotly charts.
- **Session History:** Track your progress across multiple practice sessions.
- **Enterprise PDF Reports:** Generate professional, downloadable feedback summaries including full transcripts and acoustic metrics.

### 🖥️ Hardware Telemetry
- **Live Monitoring:** Real-time sidebar dashboard showing CPU, RAM, and NVIDIA GPU/VRAM utilization.
- **OOM Protection:** Intelligent compute allocation to prevent system crashes during heavy AI inference.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Python)
- **Backend:** FastAPI (Uvicorn)
- **AI/ML:** Faster-Whisper, PyTorch, Librosa
- **Database:** Local JSON-based Session Management
- **Containerization:** Docker & Docker Compose (NVIDIA Container Toolkit supported)

---

## 📦 Installation & Setup

### Option 1: Local Development (Conda)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Karan-1708/AI-Powered-Interview-Coaching-System.git
   cd ai-interview-coach
   ```
2. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate ai-interview-coach
   ```
3. **Run the Backend API:**
   ```bash
   python src/api/server.py
   ```
4. **Run the Streamlit UI:**
   ```bash
   streamlit run app.py
   ```

### Option 2: Docker Compose (Recommended for GPU)
Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
```bash
docker-compose up --build
```

---

## 🔐 Security & Privacy
- **Local First:** All audio recordings and transcripts are stored in `./temp_data` and can be wiped instantly using the **"Delete All Data"** button in the UI.
- **Internal API Security:** Secure communication between the UI and API using an internal `X-Internal-Key` header.
- **No Data Harvesting:** Your interview data never leaves your machine unless you explicitly configure an External API provider.

---

## 🧪 Testing
The system includes a comprehensive `pytest` suite:
```bash
$env:PYTHONPATH = "."; pytest tests/
```

---

## 📜 License
This project is source-available under the **PolyForm Noncommercial License 1.0.0**. 
- **Copyright Holders:** Karanveer Singh, Amel Korandippillil Sunil, Orlando Santiago Cardenas Vargas.
- **Usage:** Free for non-commercial and educational use with prior permission.

---
*Built with ❤️ by the Data Drifters Team.*
