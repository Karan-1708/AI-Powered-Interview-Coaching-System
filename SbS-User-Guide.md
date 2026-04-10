# 🎙️ Step-by-Step User Guide: AI Interview Coach

Welcome to your personal AI Interview Coach! This guide will walk you through downloading, setting up, and mastering the application to help you ace your next big interview.

---

## 🛠️ Part 1: Getting the Application

### Step 1: Copy the Repository Link
First, you need to get the "address" of the code from GitHub.
1.  Go to the repository page on GitHub.
2.  Click the green **"<> Code"** button.
3.  Copy the URL shown in the box.

> **[INSERT SCREENSHOT: GitHub repository page highlighting the green Code button and the copied URL]**

### Step 2: Clone or Download
*   **For Tech-Savvy Users**: Open your terminal and type `git clone ` followed by the link you just copied.
*   **For Everyone Else**: In that same green **"Code"** menu, click **"Download ZIP"**. Extract the downloaded folder to a safe place on your computer (like your Desktop).

> **[INSERT SCREENSHOT: The GitHub Code menu with "Download ZIP" highlighted]**

---

## 🚀 Part 2: One-Click Setup & Launch

### Step 3: Install Prerequisites
Before clicking start, ensure your computer has these two tools:
1.  **Python**: Download from [python.org](https://www.python.org/downloads/). *Crucial: Check the box that says "Add Python to PATH" during installation.*
2.  **FFmpeg**: This is the "Ears" of the AI.
    *   **Windows**: The startup script will try to handle this for you!
    *   **Mac**: Open your terminal and type `brew install ffmpeg`.

> **[INSERT SCREENSHOT: Windows Python Installation screen with "Add Python to PATH" circled in red]**

### Step 4: Configure Security (The .env file)
To keep the application secure, you need a configuration file:
1.  Inside the project folder, find the file named **`.env.example`**.
2.  Rename it to exactly **`.env`**.
3.  Open it with Notepad and ensure the `INTERNAL_API_KEY` matches your preference (the default is fine for local use).

> **[INSERT SCREENSHOT: The project folder showing the renamed .env file]**

### Step 5: Launch the Application
Now for the magic!
*   **Windows**: Double-click the file named **`start.bat`**.
*   **Mac/Linux**: Open your terminal in the folder and run `bash start.sh`.

**What to expect**: Two windows will open. One is the "Backend" (the brain), and the other is the "Frontend" (the dashboard you will use).

> **[INSERT SCREENSHOT: The Windows terminal window running the startup script]**

---

## 🎯 Part 3: Using the Dashboard

Once the app launches, your web browser will open to `http://localhost:8501`.

### A. The Sidebar (The Control Center)
The sidebar on the left handles all the "under-the-hood" settings.

1.  **Connection Status**: Look here first! You want to see two green lights:
    *   **🟢 Backend: Online**
    *   **🟢 AI Engine: Ready**
2.  **Resource Usage**: Shows how much of your computer's power (CPU, RAM, GPU) the AI is using.
3.  **Compute Allocation**: 
    *   Choose **NVIDIA GPU** if you have a gaming graphics card (Fastest).
    *   Choose **CPU & RAM Core** for standard laptops.
4.  **Coach Voice**: Toggle between a **Male** or **Female** voice for your interviewer.
5.  **Inference Provider**:
    *   **Ollama (Local)**: 100% private. Use this if you have [Ollama](https://ollama.com/) installed.
    *   **External API**: Use this for high-end cloud models like OpenAI or Google Gemini (Requires an API Key).
6.  **Download New Model**: Use this to download fresh AI models directly to your computer.
7.  **🔄 Start New Interview**: Resets your chat but keeps your settings.
8.  **🗑️ Danger Zone**: Permanently deletes all your history and recordings.

> **[INSERT SCREENSHOT: The full Sidebar with all 8 sections numbered]**

---

### B. The Setup Wizard (Defining your Role)
Before the interview starts, the AI needs to know who you are.

1.  **Define Role**: Enter your target Industry, Job Title, and Seniority.
2.  **Upload Context (Highly Recommended)**: 
    *   Upload your **Resume**. The AI will scan it and ask questions about *your* specific experience.
    *   Upload the **Job Description**. The AI will act like it's hiring specifically for that role.
3.  **Generate Rounds**: Click this to see the 4 interview stages the AI has designed for you.
4.  **Select Stage & Style**: Choose which round you want to practice (e.g., Technical Assessment) and how tough you want the interviewer to be.

> **[INSERT SCREENSHOT: The Setup Wizard with a Resume uploaded and rounds generated]**

---

### C. The Live Interview Simulator
This is where you practice!

1.  **The Greeting**: The AI will introduce itself (sometimes impersonating the recruiter from your job posting!) and greet you by name.
2.  **Recording**: 
    *   Click the **microphone icon** to start speaking.
    *   Click the **stop button** when you are finished.
3.  **Submit**: Click **"Submit Answer"**. The AI will listen, analyze your words, and ask a follow-up question.
4.  **🔊 Replay**: If you missed a question, click the Replay button under the AI's message to hear it again.

> **[INSERT SCREENSHOT: The chat interface showing an AI question and the recording widget]**

---

### D. Final Analysis & Reports
When you're ready, click **"End Interview & Analyze"**.

1.  **AI Feedback**: A deep-dive evaluation of your performance using the professional **STAR Framework**.
2.  **Acoustic Metrics**: See your **Words Per Minute (WPM)** and how many "filler words" (um, uh, like) you used.
3.  **PDF Export**: Click **"Download PDF Report"** to get a professional summary you can save or share.

> **[INSERT SCREENSHOT: The Final Analysis dashboard with the PDF download button visible]**

---

### E. Session History
Toggle to the **"📈 Session History"** tab at the top to track your long-term growth.

1.  **Performance Badges**: Earn badges like "Master Interviewer" or "Golden Pacer" as you improve.
2.  **Progress Charts**: View graphs of your pacing and fluency over time to see exactly where you are getting better.

> **[INSERT SCREENSHOT: The Session History tab showing badges and Plotly charts]**

---
*Created by the Data Drifters Team.*
