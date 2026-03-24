import os
import platform
import shutil
import streamlit as st
import logging
import requests
import ast
import time
import pandas as pd
import plotly.express as px

# --- 1. GLOBAL CRASH PROTECTION ---
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.ui.recorder import record_audio
from src.backend.hardware import HardwareInfo
from src.backend.monitor import ResourceMonitor
from src.utils.diagnostics import log_system_info, get_logger
from src.utils.file_manager import FileManager
from src.utils.pdf_generator import PDFGenerator

# Initialize Logging
log_system_info()
logger = get_logger()

# ==========================================
# THE API CLIENT (THE BRIDGE)
# ==========================================
class APIClient:
    # If the environment variable isn't set (Local), use localhost. If it is (Docker), use 'api'.
    BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

    @staticmethod
    def get_hardware_status():
        """Fetches live telemetry from the backend API container."""
        url = f"{APIClient.BASE_URL}/hardware"
        try:
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                return res.json()
        except: pass
        return None

    @staticmethod
    def process_audio(file_path, difficulty, tier):
        """Sends the audio file to the backend API for Whisper transcription."""
        url = f"{APIClient.BASE_URL}/process-audio"
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "audio/wav")}
                data = {"difficulty": difficulty, "tier": tier}
                response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                res = response.json()
                return res['transcript'], res['metrics'], res['duration'], None
            return None, None, None, f"API Error: {response.text}"
        except requests.exceptions.ConnectionError:
            return None, None, None, "Connection Error: Is the FastAPI server running?"

    @staticmethod
    def generate_response(system_prompt, user_message, chat_history, tier):
        """Sends the chat history to the backend API for Ollama inference."""
        url = f"{APIClient.BASE_URL}/generate-response"
        payload = {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "chat_history": chat_history,
            "tier": tier
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()['response']
            return f"API Error: {response.text}"
        except requests.exceptions.ConnectionError:
            return "Connection Error: Is the FastAPI server running?"

# --- DLL FIX ---
def register_nvidia_dlls():
    if platform.system() != "Windows": return
    try:
        import site
        possible_paths = site.getsitepackages()
        try: possible_paths.append(site.getusersitepackages())
        except: pass
        for base_path in possible_paths:
            cublas_bin = os.path.join(base_path, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(base_path, "nvidia", "cudnn", "bin")
            if os.path.exists(os.path.join(cublas_bin, "cublas64_12.dll")):
                os.environ["PATH"] += os.pathsep + cublas_bin
                os.environ["PATH"] += os.pathsep + cudnn_bin
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(cublas_bin)
                    os.add_dll_directory(cudnn_bin)
                break
    except Exception: pass
register_nvidia_dlls()

# --- LIVE HARDWARE FRAGMENT ---
@st.fragment(run_every=2)
def live_hardware_monitor(selected_tier):
    hw_status = APIClient.get_hardware_status()
    if not hw_status:
        # Changed from st.sidebar.warning to st.warning
        st.warning("⚠️ Cannot connect to API Telemetry.")
        return
        
    stats = hw_status["stats"]
    has_nvidia = hw_status["has_nvidia"]
    
    if "Pro" in selected_tier and has_nvidia:
        st.progress(stats['vram_percent'] / 100, text=f"API VRAM: {stats['vram_used_gb']}/{stats['vram_total_gb']} GB")
    else:
        st.progress(stats['cpu_percent'] / 100, text=f"API CPU: {stats['cpu_percent']}%")
        st.progress(stats['ram_percent'] / 100, text=f"API RAM: {stats['ram_used_gb']}/{stats['ram_total_gb']} GB")

# --- MAIN APP ---
st.set_page_config(page_title="AI Interview Coach", page_icon="🎙️", layout="wide")

def main():
    try:
        st.title("🎙️ AI Interview Coach")
        
        # --- SIDEBAR ---
        st.sidebar.header("🖥️ API Hardware Monitor")
        
        # Pull hardware data from the backend!
        hw_status = APIClient.get_hardware_status()
        if hw_status:
            rec_tier = hw_status["tier"]
            rec_reason = hw_status["reason"]
        else:
            rec_tier, rec_reason = "Eco (Low Spec)", "🔴 API Offline or Booting..."
        
        default_index = 0
        if "Balanced" in rec_tier: default_index = 1
        if "Pro" in rec_tier: default_index = 2
        
        selected_tier = st.sidebar.selectbox("Performance Profile", 
            ["Eco (Low Spec)", "Balanced (Mid Spec)", "Pro (High Spec)"], 
            index=default_index, help=f"Recommendation: {rec_reason}")

        with st.sidebar:
            live_hardware_monitor(selected_tier) # Pass only the tier now

        st.sidebar.divider()
        st.sidebar.header("🔒 Privacy & Data")
        
        audio_files = len(os.listdir(FileManager.TEMP_DIR)) if os.path.exists(FileManager.TEMP_DIR) else 0
        log_files = len(os.listdir(FileManager.LOG_DIR)) if os.path.exists(FileManager.LOG_DIR) else 0
        
        st.sidebar.caption(f"Stored Data: {audio_files} recordings, {log_files} logs.")
        
        # We removed the 'if files > 0' check so you can force a reset anytime
        if st.sidebar.button("🗑️ Hard Reset & Delete All Data", type="primary"):
            # 1. Delete physical audio and log files
            count = FileManager.cleanup_all_data() 
            
            # 2. Delete the History JSON file
            try:
                from src.utils.history import HistoryManager
                HistoryManager.clear_history()
            except Exception as e: 
                logger.error(f"Failed to clear history: {e}")

            # 3. Nuke the Streamlit Session State (The Virtual RAM)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
                
            st.sidebar.success("System wiped. Rebooting interface...")
            time.sleep(1) # Give the user 1 second to read the success message
            st.rerun()

        st.sidebar.divider()

        # --- MAIN INTERFACE ---
        tab_coach, tab_history = st.tabs(["🎯 Live Coach", "📈 Session History"])
        
        with tab_coach:
            # Initialize Session States
            if 'setup_step' not in st.session_state: st.session_state['setup_step'] = 1
            if 'rounds' not in st.session_state: st.session_state['rounds'] = []
            if 'custom_questions' not in st.session_state: st.session_state['custom_questions'] = []
            if 'round_info' not in st.session_state: st.session_state['round_info'] = {}
            
            # --- WIZARD STEP 1 & 2: CONTEXT & ROUND SELECTION ---
            with st.expander("🛠️ Interview Setup Wizard", expanded=(st.session_state['setup_step'] < 3)):
                st.markdown("### 1. Define Your Target Role")
                
                col_ind, col_role, col_sen = st.columns(3)
                industry = col_ind.text_input("Industry / Field", placeholder="e.g., Tech, Finance")
                job_title = col_role.text_input("Job Title", placeholder="e.g., Backend Developer")
                seniority = col_sen.selectbox("Seniority Level", ["Entry-Level", "Mid-Level", "Senior / Lead", "Executive"])
                
                if st.button("Generate Interview Rounds", disabled=not (industry and job_title)):
                    with st.spinner("🧠 API is structuring the interview process..."):
                        
                        prompt = f"You are an expert tech recruiter. List 4 realistic interview rounds for a {seniority} {job_title} in the {industry} industry. Output ONLY a Python-style list of strings, nothing else. Example: ['1. HR Screen', '2. Technical']"
                        response = APIClient.generate_response(
                            system_prompt="You output strictly formatted lists.", 
                            user_message=prompt, 
                            chat_history=[], 
                            tier=selected_tier
                        )
                        
                        try:
                            st.session_state['rounds'] = ast.literal_eval(response)
                        except:
                            st.session_state['rounds'] = [r.strip() for r in response.replace('[', '').replace(']', '').replace("'", "").split(',')]
                            
                        st.session_state['setup_step'] = 2
                        st.rerun()

                if st.session_state['setup_step'] >= 2:
                    st.divider()
                    st.markdown("### 2. Select Interview Stage")
                    selected_round = st.selectbox("Which round are you preparing for?", st.session_state['rounds'])
                    
                    if st.button("Generate Custom Questions", type="primary"):
                        with st.spinner(f"🧠 API is writing questions for the {selected_round}..."):
                            
                            round_type = selected_round.split(" ")[0]
                            
                            if "HR" in round_type or "Screen" in round_type or "First" in round_type:
                                meaning = "A standard, efficient first-round interview. Focus on high-level experience and culture fit."
                                rec_mode = "Standard Interview"
                                rec_persona = "🤝 Friendly HR Recruiter (Focuses on soft skills & culture fit)"
                            elif "Technical" in round_type or "System" in round_type or "Code" in round_type:
                                meaning = "Common for technical assessments. Expect in-depth scrutiny and follow-ups."
                                rec_mode = "Technical / Complex"
                                rec_persona = "💼 Strict Technical Lead (Focuses purely on accuracy & efficiency)"
                            else: 
                                meaning = "Advanced panel or final stage interview. High pressure."
                                rec_persona = "🔥 Stress Interviewer (Highly critical, looks for flaws & hesitations)"
                                rec_mode = "Presentation" if seniority == "Executive" else "Technical / Complex"

                            st.session_state['round_info'] = {
                                "meaning": meaning, 
                                "recommended_mode": rec_mode,
                                "recommended_persona": rec_persona
                            }
                            
                            q_prompt = f"Generate 3 highly specific interview questions for a {seniority} {job_title} during the '{selected_round}' round. Output ONLY a Python-style list of strings."
                            q_response = APIClient.generate_response(
                                system_prompt="You are an expert interviewer. You output strictly formatted lists.", 
                                user_message=q_prompt, 
                                chat_history=[], 
                                tier=selected_tier
                            )
                            
                            try:
                                questions = ast.literal_eval(q_response)
                            except:
                                questions = [q.strip() for q in q_response.replace('[', '').replace(']', '').replace("'", "").split('\n') if q.strip()]
                                
                            questions.append("-- Custom Question --")
                            st.session_state['custom_questions'] = questions
                            
                            st.session_state['setup_step'] = 3
                            st.rerun()

            # --- WIZARD STEP 3: THE CONVERSATIONAL LOOP ---
            if st.session_state['setup_step'] == 3:
                st.subheader("🎙️ Live Interview Simulator")
                
                info = st.session_state['round_info']
                st.info(f"⏱️ **Stage Context:** {info['meaning']} | **Interviewer:** {info['recommended_persona']}")

                if 'chat_history' not in st.session_state:
                    sys_prompt = f"You are acting as a {info['recommended_persona']} conducting a {selected_round} interview for a {seniority} {job_title} role in the {industry} industry. Your goal is to assess the candidate's skills based on the persona. Ask ONE question at a time. Keep your questions concise (1-2 sentences). Base your follow-ups strictly on the candidate's previous answer. Do not break character. Do not provide feedback yet."
                    
                    st.session_state['system_prompt'] = sys_prompt
                    st.session_state['chat_history'] = []
                    st.session_state['aggregated_metrics'] = [] 
                    
                    first_q = st.session_state['custom_questions'][0]
                    st.session_state['chat_history'].append({"role": "assistant", "content": first_q})
                    
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(first_q)
                        engine.runAndWait()
                    except: pass

                st.divider()
                for msg in st.session_state['chat_history']:
                    avatar = "🤖" if msg["role"] == "assistant" else "👤"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])

                st.divider()
                st.markdown("### Your Response")
                input_method = st.radio("Input Method", ["🎙️ Record Live", "📁 Upload Audio"], horizontal=True, label_visibility="collapsed")
                
                audio_path = None
                if input_method == "🎙️ Record Live":
                    audio_path = record_audio()
                else:
                    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
                    if uploaded_file:
                        audio_path = os.path.join(FileManager.TEMP_DIR, uploaded_file.name)
                        with open(audio_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                col_submit, col_end = st.columns(2)
                
                with col_submit:
                    if audio_path and st.button("🗣️ Submit Answer", type="primary", width="stretch"):
                        with st.spinner("Transcribing and processing..."):
                            transcript, metrics, duration, error = APIClient.process_audio(audio_path, info['recommended_mode'], selected_tier)
                            
                            if error:
                                st.error(f"⚠️ {error}")
                            else:
                                st.session_state['aggregated_metrics'].append({
                                    "transcript": transcript,
                                    "metrics": metrics,
                                    "duration": duration
                                })
                                st.session_state['chat_history'].append({"role": "user", "content": transcript})
                                
                                with st.spinner("🧠 API is thinking..."):
                                    next_question = APIClient.generate_response(
                                        system_prompt=st.session_state['system_prompt'], 
                                        user_message=transcript, 
                                        chat_history=st.session_state['chat_history'][:-1], 
                                        tier=selected_tier
                                    )
                                    
                                    st.session_state['chat_history'].append({"role": "assistant", "content": next_question})
                                    
                                    try:
                                        engine = pyttsx3.init()
                                        engine.say(next_question)
                                        engine.runAndWait()
                                    except: pass
                                    
                                    st.rerun()

                with col_end:
                    if len(st.session_state['chat_history']) > 1:
                        if st.button("🛑 End Interview & Analyze", width="stretch"):
                            if audio_path:
                                with st.spinner("Processing final answer..."):
                                    transcript, metrics, duration, error = APIClient.process_audio(audio_path, info['recommended_mode'], selected_tier)
                                    if not error:
                                        st.session_state['aggregated_metrics'].append({
                                            "transcript": transcript,
                                            "metrics": metrics,
                                            "duration": duration
                                        })
                                        st.session_state['chat_history'].append({"role": "user", "content": transcript})
                            
                            st.session_state['interview_complete'] = True
                            st.rerun()

            # --- WIZARD STEP 4: THE GRAND FINALE (DASHBOARD) ---
            if st.session_state.get('interview_complete', False):
                st.divider()
                st.header("📊 Final Interview Analysis")
                
                if 'final_feedback' not in st.session_state:
                    with st.spinner("🧠 API is compiling your final STAR feedback..."):
                        total_wpm = 0
                        total_fillers = 0
                        total_duration = 0
                        valid_turns = len(st.session_state['aggregated_metrics'])
                        
                        if valid_turns > 0:
                            for turn in st.session_state['aggregated_metrics']:
                                m = turn['metrics']
                                total_wpm += m['wpm']
                                total_fillers += m['filler_count']
                                total_duration += turn['duration']
                            avg_wpm = total_wpm / valid_turns
                        else:
                            avg_wpm = 0
                            
                        full_transcript = ""
                        for msg in st.session_state['chat_history']:
                            speaker = "Interviewer" if msg['role'] == "assistant" else "Candidate"
                            full_transcript += f"<b>{speaker}:</b> {msg['content']}<br><br>\n"
                            
                        # 2. Strict Markdown Headers (No Numbers!)
                        final_prompt = f"""
                        You are a senior hiring manager. Review this interview transcript for a {seniority} {job_title} role in the {industry} industry.
                        
                        TRANSCRIPT:
                        {full_transcript}
                        
                        Provide a comprehensive evaluation. Format your response strictly using these Markdown headers (Do NOT use numbered lists for the section titles):
                        
                        ### Overall Impression
                        (1 short paragraph)
                        
                        ### Key Strengths
                        (Use standard bullet points)
                        
                        ### Areas for Improvement
                        (Use standard bullet points)
                        
                        ### STAR Framework Analysis
                        (Short paragraph evaluating if they used Situation, Task, Action, Result)
                        """
                        
                        final_feedback = APIClient.generate_response(
                            system_prompt="You are an expert, direct, and highly constructive career coach.", 
                            user_message=final_prompt, 
                            chat_history=[], 
                            tier=selected_tier
                        )
                        
                        try:
                            from src.utils.history import HistoryManager
                            HistoryManager.save_session(avg_wpm, total_fillers, "Multi-Turn", info['recommended_mode'])
                        except: pass
                        
                        st.session_state['final_feedback'] = final_feedback
                        st.session_state['full_transcript'] = full_transcript
                        st.session_state['avg_wpm'] = avg_wpm
                        st.session_state['total_fillers'] = total_fillers
                        st.session_state['total_duration'] = total_duration

                tab_feedback, tab_metrics, tab_transcript = st.tabs(["🧠 AI Coach Feedback", "📈 Acoustic Metrics", "📝 Full Transcript"])
                
                with tab_feedback:
                    st.markdown(st.session_state['final_feedback'])
                    
                with tab_metrics:
                    col1, col2, col3 = st.columns(3)
                    mem_wpm = st.session_state['avg_wpm']
                    mem_fillers = st.session_state['total_fillers']
                    mem_duration = st.session_state['total_duration']
                    
                    wpm_delta = "Ideal Pace" if 130 <= mem_wpm <= 160 else "Too Fast/Slow"
                    wpm_color = "normal" if 130 <= mem_wpm <= 160 else "inverse"
                    
                    col1.metric("Average Pacing", f"{mem_wpm:.0f} WPM", delta=wpm_delta, delta_color=wpm_color)
                    col2.metric("Total Filler Words", mem_fillers)
                    col3.metric("Total Speaking Time", f"{mem_duration:.1f}s")
                    st.info("💡 Note: A conversational pace of 130-160 WPM is considered highly confident and professional.")
                    
                with tab_transcript:
                    st.markdown(st.session_state['full_transcript'])
                    
                # --- PDF REPORT GENERATOR ---
                st.divider()
                
                # Bundle the metrics
                metrics_data = {
                    'wpm': mem_wpm,
                    'fillers': mem_fillers,
                    'duration': mem_duration
                }
                
                # Define a safe temporary path for the PDF
                pdf_filename = f"report_{int(time.time())}.pdf"
                pdf_path = os.path.join(FileManager.TEMP_DIR, pdf_filename)
                
                # Generate the file
                success = PDFGenerator.generate_report(
                    job_title, industry, metrics_data, 
                    st.session_state['final_feedback'], 
                    st.session_state['full_transcript'], 
                    pdf_path
                )
                
                if success:
                    # Read the generated PDF into memory for Streamlit to serve
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        
                    st.download_button(
                        label="📄 Download Enterprise PDF Report",
                        data=pdf_bytes,
                        file_name=f"Data_Drifters_Report_{job_title.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        width="stretch"
                    )
                else:
                    st.error("⚠️ Failed to generate PDF report. Check system logs.")

        # ==========================================
        # GAMIFICATION & HISTORY DASHBOARD
        # ==========================================
        with tab_history:
            st.header("📈 Your Coaching Progress")
            
            # Load history using the HistoryManager
            try:
                from src.utils.history import HistoryManager
                history_data = HistoryManager.load_history()
            except ImportError:
                history_data = []

            if not history_data:
                st.info("No session history found. Complete your first practice interview to see your progress here!")
            else:
                # Convert the JSON list to a Pandas DataFrame for easy graphing
                df = pd.DataFrame(history_data)

                # --- TOP LEVEL METRICS ---
                total_sessions = len(df)
                avg_wpm_all = df['wpm'].mean()
                total_fillers_all = df['fillers'].sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Practice Sessions", total_sessions)
                col2.metric("All-Time Average Pace", f"{avg_wpm_all:.0f} WPM")
                col3.metric("Total Fillers Tracked", total_fillers_all)

                st.divider()

                # --- PLOTLY INTERACTIVE CHARTS ---
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.subheader("🗣️ Pacing Over Time (WPM)")
                    # Draw a line chart for WPM
                    fig_wpm = px.line(
                        df, x='timestamp', y='wpm', markers=True,
                        title="Words Per Minute (Target: 130-160)",
                        labels={"timestamp": "Session Date", "wpm": "WPM"}
                    )
                    # Add a green "Ideal Range" band in the background!
                    fig_wpm.add_hrect(y0=130, y1=160, line_width=0, fillcolor="green", opacity=0.1)
                    fig_wpm.update_yaxes(tickformat="d")
                    st.plotly_chart(fig_wpm, width='stretch')

                with col_chart2:
                    st.subheader("🛑 Filler Words Over Time")
                    # Draw a bar chart for Filler Words, colored by severity
                    fig_fillers = px.bar(
                        df, x='timestamp', y='fillers',
                        title="Total Filler Words (Um, Uh, Like)",
                        labels={"timestamp": "Session Date", "fillers": "Filler Count"},
                        color='fillers', color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig_fillers, width='stretch')

                st.divider()

                # --- RAW DATA TABLE ---
                st.subheader("📝 Raw Session Logs")
                st.dataframe(df, width='stretch', hide_index=True)

    except Exception as e:
        st.error("🚨 An unexpected error occurred.")
        st.code(str(e))
        logger.critical(f"Global Crash: {e}", exc_info=True)

if __name__ == "__main__":
    FileManager.initialize_directories()
    main()