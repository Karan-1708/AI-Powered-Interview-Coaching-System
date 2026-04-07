# 📖 User Guide: AI Interview Coach

Welcome to the **AI Interview Coach**! This guide will help you get the application running on your computer with just a few clicks.

---

## 🚀 One-Click Setup

### 1. Prerequisites
Before starting, ensure you have the following installed:
*   **Python (3.10 to 3.13)**: Download from [python.org](https://www.python.org/downloads/). *Important: Check "Add Python to PATH" during installation.*
*   **FFmpeg**: Required for the AI to "hear" you.
    *   **Windows**: [Download here](https://ffmpeg.org/download.html).
    *   **Mac**: Open Terminal and type `brew install ffmpeg`.

### 2. Launching the App
1.  **Download & Extract** the project folder.
2.  **Windows Users**: Double-click the `start.bat` file.
3.  **Mac/Linux Users**: Open a terminal in the folder and run `bash start.sh`.

### 3. What the Script Does
*   Creates a private virtual environment so it doesn't mess with your computer.
*   Installs all necessary AI components (including GPU support if you have an NVIDIA card).
*   Launches the **Backend Engine** and the **Frontend Dashboard** automatically.

---

## 🎯 How to Use the Coach

### Step 1: Configuration
*   **Sidebar**: Choose your **Coach Voice** (Male/Female).
*   **Inference Provider**: 
    *   Select **Local (Ollama)** for 100% private, offline coaching (Requires [Ollama](https://ollama.com/) to be installed).
    *   Select **External API** (OpenAI, Gemini, or Claude) for the highest quality coaching (Requires your own API Key).

### Step 2: Setup your Interview
1.  Enter your **Industry**, **Job Title**, and **Seniority**.
2.  **Upload Context (Optional)**: Upload your Resume or the Job Description. The AI will use these to ask highly personalized questions.
3.  Click **Generate Interview Rounds** to see your career path.
4.  Select a **Round** (e.g., Technical Round) and an **Interviewer Style**.

### Step 3: The Interview
1.  The Coach will greet you (by name if it's in your resume!) and ask the first question.
2.  Click the **Record** button to speak your answer.
3.  Click **Submit Answer** to send it to the coach.
4.  When finished, click **End Interview & Analyze**.

### Step 4: Performance Review
*   View your **Overall Impression** and **Key Strengths**.
*   Check your **Acoustic Metrics**: See your Words Per Minute (WPM) and how many "filler words" (um, uh, like) you used.
*   **Export**: Download your full feedback and transcript as a professional PDF.

---

## 🔐 Privacy & Safety
*   **Your Data stays Local**: All audio recordings and transcripts are stored on your own computer in the `temp_data` folder.
*   **Instant Wipe**: Use the **"Delete All Data"** button in the sidebar to permanently erase all session history and recordings.
