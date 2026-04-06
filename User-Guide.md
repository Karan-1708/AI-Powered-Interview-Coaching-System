# Setup & Installation Guide

This project is designed to be highly portable. While we support various environments, the **standard Python Virtual Environment (venv)** is the recommended way to set up the system.

---

## 1. System Requirements (Prerequisites)

Before setting up the Python environment, you must have the following installed on your system:

### 1.1 Audio Processing (Required)

- **FFmpeg**: Essential for AI transcription.
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your System PATH. 
  - **Mac**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

### 1.2 GPU Acceleration (Optional but Recommended)

- **NVIDIA Users**: Ensure you have the latest NVIDIA drivers installed to enable GPU acceleration.
- **Apple Silicon Users**: No extra drivers needed; the system uses the Apple Neural Engine automatically.

---

## 2. Primary Setup: Python Virtual Environment (Recommended)

This method works on any system with Python 3.10 - 3.13 installed.

### Step 1: Create the Environment

Open your terminal in the project root folder and run:

```bash
# Create a virtual environment named '.venv'
python -m venv .venv
```

### Step 2: Activate the Environment

- **Windows**: `.\venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### Step 3: Install High-Performance AI Engine (CUDA)

To ensure the system uses your GPU (NVIDIA), run this specific command first:

```bash
# For Python 3.13 (Windows)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# For Python 3.10 - 3.12 (Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Alternative Setup: Anaconda / Miniconda

If you prefer using Conda, follow these steps:

```bash
# Create environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate ai-interview-coach
```

---
## 4. Troubleshooting Hardware Detection

If the backend logs show `CUDA: NOT DETECTED`:

1. Verify your installation by running: `pip list`
2. Ensure `torch` has a `+cuXXX` suffix (e.g., `2.7.0.dev+cu124`).
3. If it shows a plain version number, the CPU version was installed by mistake. Re-run **Step 3** from the Primary Setup above.

---
## 5. Launching the Program
To ensure all components communicate correctly, you must launch them in the following order:

### Step 1: Start Ollama (For Local AI & Privacy)
Ensure [Ollama](https://ollama.com/) is installed and running on your system. This is required if you want to use the **Local (Ollama)** provider for 100% private, offline coaching.

### Step 2: Start the Backend (FastAPI)
Open a terminal, activate your environment (`venv` or `conda`), and run:
```bash
# From the project root
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Start the Frontend (Streamlit)
Open a **second** terminal, activate your environment, and run:
```bash
# From the project root
streamlit run app.py
```
