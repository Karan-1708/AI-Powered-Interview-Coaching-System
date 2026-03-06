import os
import platform
import shutil
import streamlit as st  # pyright: ignore[reportMissingImports]
import logging
import pandas as pd
from src.utils.history import HistoryManager
from src.ui.dashboard import render_dashboard

# --- 1. GLOBAL CRASH PROTECTION ---
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.ui.recorder import record_audio
from src.backend.audio_processor import AudioProcessor
from src.backend.hardware import HardwareInfo
from src.backend.monitor import ResourceMonitor
from src.utils.diagnostics import log_system_info, get_logger

# Initialize Logging
log_system_info()
logger = get_logger()

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

# --- HELPER: PRIVACY CLEANUP ---
def cleanup_data():
    """Deletes all temporary files (Privacy Feature) - Windows Safe Version"""
    deleted_files = 0
    
    # 1. Clean Audio Files
    if os.path.exists("temp_data"):
        for f in os.listdir("temp_data"):
            try:
                os.remove(os.path.join("temp_data", f))
                deleted_files += 1
            except Exception:
                pass # Skip if file is actively being recorded

    # 2. Clean Log Files (Critical Fix for WinError 32)
    if os.path.exists("logs"):
        # Step A: Get the logger and find the file handler
        logger = logging.getLogger()
        
        # Step B: Close and remove all handlers to release the Windows file lock
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        
        # Step C: Now it is safe to delete
        for f in os.listdir("logs"):
            try:
                os.remove(os.path.join("logs", f))
                deleted_files += 1
            except Exception as e:
                st.error(f"Could not delete {f}: {e}")
            
    return deleted_files

# --- MAIN APP ---
st.set_page_config(page_title="AI Interview Coach", page_icon="🎙️", layout="wide")

# --- NEW: LIVE HARDWARE FRAGMENT ---
@st.fragment(run_every=2)
def live_hardware_monitor(monitor, hw, selected_tier):
    stats = monitor.get_system_usage()
    
    # We use regular st.progress here, the sidebar context will handle placement
    if "Pro" in selected_tier and hw.has_nvidia:
        st.progress(stats['vram_percent'] / 100, text=f"VRAM: {stats['vram_used_gb']}/{stats['vram_total_gb']} GB")
    else:
        st.progress(stats['cpu_percent'] / 100, text=f"CPU: {stats['cpu_percent']}%")
        st.progress(stats['ram_percent'] / 100, text=f"RAM: {stats['ram_used_gb']}/{stats['ram_total_gb']} GB")

def main():
    try:
        hw = HardwareInfo()
        monitor = ResourceMonitor()
        
        st.title("🎙️ AI Interview Coach")
        
        # --- SIDEBAR (Keep your existing sidebar logic here) ---
        st.sidebar.header("🖥️ Hardware Monitor")
        rec_tier, rec_reason = hw.get_recommendation()
        
        default_index = 0
        if "Balanced" in rec_tier: default_index = 1
        if "Pro" in rec_tier: default_index = 2
        
        selected_tier = st.sidebar.selectbox("Performance Profile", 
            ["Eco (Low Spec)", "Balanced (Mid Spec)", "Pro (High Spec)"], 
            index=default_index, help=f"Recommendation: {rec_reason}")

        with st.sidebar:
            live_hardware_monitor(monitor, hw, selected_tier)

        st.sidebar.divider()
        st.sidebar.header("🔒 Privacy & Data")
        
        audio_files = len(os.listdir("temp_data")) if os.path.exists("temp_data") else 0
        log_files = len(os.listdir("logs")) if os.path.exists("logs") else 0
        
        st.sidebar.caption(f"Stored Data: {audio_files} recordings, {log_files} logs.")
        if (audio_files + log_files) > 0:
            if st.sidebar.button("🗑️ Delete All Data", type="primary"):
                count = cleanup_data()
                st.sidebar.success(f"Deleted {count} files.")
                st.rerun()

        st.sidebar.divider()

        # --- MAIN INTERFACE (TABS OVERHAUL) ---
        tab_coach, tab_history = st.tabs(["🎯 Live Coach", "📈 Session History"])
        
        with tab_coach:
            processor = AudioProcessor()
            
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
                    with st.spinner("🧠 AI is structuring the interview process..."):
                        st.session_state['rounds'] = [
                            "Recruiter/Phone Screening (15–30 min)",
                            "First-Round/Hiring Manager (30–60 min)",
                            "Technical/In-depth Interview (60–90 min)",
                            "Panel/Group Interviews (60–90+ min)",
                            "Final Interview (30–60+ min)"
                        ]
                        st.session_state['setup_step'] = 2
                        st.rerun()

                if st.session_state['setup_step'] >= 2:
                    st.divider()
                    st.markdown("### 2. Select Interview Stage")
                    selected_round = st.selectbox("Which round are you preparing for?", st.session_state['rounds'])
                    
                    if st.button("Generate Custom Questions", type="primary"):
                        with st.spinner(f"🧠 AI is writing questions for the {selected_round.split('(')[0]}..."):
                            
                            round_type = selected_round.split(" ")[0]
                            
                            # --- NEW: AUTOMATED MAPPING LOGIC ---
                            if "Recruiter" in round_type or "First-Round" in round_type:
                                meaning = "A standard, efficient first-round interview. Focus on high-level experience and culture fit."
                                rec_mode = "Standard Interview"
                                rec_persona = "🤝 Friendly HR Recruiter (Focuses on soft skills & culture fit)"
                            elif "Technical" in round_type:
                                meaning = "Common for technical assessments. Expect in-depth scrutiny and follow-ups."
                                rec_mode = "Technical / Complex"
                                rec_persona = "💼 Strict Technical Lead (Focuses purely on accuracy & efficiency)"
                            elif "Panel" in round_type:
                                meaning = "Panel interviews involve multiple stakeholders. High pressure, varied question types."
                                rec_mode = "Technical / Complex"
                                rec_persona = "🔥 Stress Interviewer (Highly critical, looks for flaws & hesitations)"
                            else: # Final Interview
                                meaning = "Final interviews evaluate ultimate culture fit, long-term alignment, and leadership."
                                rec_persona = "🔥 Stress Interviewer (Highly critical, looks for flaws & hesitations)"
                                rec_mode = "Presentation" if seniority == "Executive" else "Standard Interview"

                            st.session_state['round_info'] = {
                                "meaning": meaning, 
                                "recommended_mode": rec_mode,
                                "recommended_persona": rec_persona
                            }
                            
                            # Mock custom questions
                            st.session_state['custom_questions'] = [
                                f"Tell me about your experience as a {seniority} {job_title}.",
                                f"What is your approach to handling {industry} challenges in a {round_type.lower()} setting?",
                                "-- Custom Question --"
                            ]
                            st.session_state['setup_step'] = 3
                            st.rerun()

            # --- WIZARD STEP 3: THE INTERVIEW SIMULATOR ---
            if st.session_state['setup_step'] == 3:
                col1, col2 = st.columns([1, 1.5])

                with col1:
                    st.subheader("3. Interview Environment")
                    
                    info = st.session_state['round_info']
                    st.info(f"⏱️ **Stage Context:** {info['meaning']}")

                    # --- NEW: AUTO-SELECTED & LOCKED DROPDOWNS ---
                    personas = [
                        "🤝 Friendly HR Recruiter (Focuses on soft skills & culture fit)",
                        "💼 Strict Technical Lead (Focuses purely on accuracy & efficiency)",
                        "🔥 Stress Interviewer (Highly critical, looks for flaws & hesitations)"
                    ]
                    modes = ["Practice Mode", "Standard Interview", "Technical / Complex", "Presentation"]
                    
                    p_idx = personas.index(info['recommended_persona']) if info['recommended_persona'] in personas else 0
                    m_idx = modes.index(info['recommended_mode']) if info['recommended_mode'] in modes else 1
                    
                    # Displayed to the user, but disabled so they can't change it
                    selected_persona = st.selectbox("Interviewer Persona (Auto-Assigned)", personas, index=p_idx, disabled=True)
                    selected_mode = st.selectbox("Analysis Mode (Auto-Assigned)", modes, index=m_idx, disabled=True)

                    st.divider()
                    st.subheader("4. Target Question")
                    
                    q_col, btn_col = st.columns([4, 1])
                    with q_col:
                        selected_q = st.selectbox("Select Question", st.session_state['custom_questions'], label_visibility="collapsed")
                    
                    target_question = selected_q
                    if selected_q == "-- Custom Question --":
                        target_question = st.text_area("Type your custom question here:")
                    
                    with btn_col:
                        if st.button("🗣️ Ask", use_container_width=True):
                            try:
                                import pyttsx3
                                engine = pyttsx3.init()
                                engine.say(target_question)
                                engine.runAndWait()
                            except Exception as e:
                                st.toast(f"Audio playback error: {e}", icon="🔇")
                    
                    st.divider()
                    st.subheader("5. Provide Answer")
                    
                    input_method = st.radio("Input Method", ["🎙️ Record Live", "📁 Upload Audio"], horizontal=True)
                    
                    audio_path = None
                    if input_method == "🎙️ Record Live":
                        audio_path = record_audio()
                        if audio_path: st.audio(audio_path)
                    else:
                        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
                        if uploaded_file:
                            audio_path = os.path.join("temp_data", uploaded_file.name)
                            with open(audio_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.audio(audio_path)

                    if audio_path:
                        if st.button(f"Analyze Answer ({selected_tier})", type="primary", use_container_width=True):
                            with st.status("Analyzing your performance...", expanded=True) as status:
                                st.write("🔍 Running Pre-Flight Silence Check...")
                                transcript, metrics, duration, error = processor.process_interview(
                                    audio_path, difficulty=selected_mode, tier=selected_tier
                                )
                                
                                if error:
                                    status.update(label="Analysis Failed", state="error", expanded=True)
                                    st.error(f"⚠️ {error}")
                                else:
                                    # Save to history tracking
                                    HistoryManager.save_session(metrics['wpm'], metrics['filler_count'], metrics['tone_label'], selected_mode)
                                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                                    
                                    # Send full context to dashboard
                                    full_context = f"[{seniority} {job_title}] - {selected_q}"
                                    st.session_state['results'] = (transcript, metrics, duration, full_context, selected_persona)
                                    st.rerun()

                with col2:
                    if 'results' in st.session_state:
                        transcript, metrics, duration, saved_q, saved_persona = st.session_state['results']
                        from src.ui.dashboard import render_dashboard
                        render_dashboard(transcript, metrics, duration, selected_mode, selected_tier, saved_q, saved_persona)
                    else:
                        st.info("Ready for analysis. Complete the setup and provide your answer.")

        # --- HISTORY TAB ---
        with tab_history:
            st.subheader("📈 Your Progression")
            history_data = HistoryManager.load_history()
            
            if history_data:
                df = pd.DataFrame(history_data)
                
                # Layout metrics
                h1, h2, h3 = st.columns(3)
                h1.metric("Total Sessions", len(df))
                h2.metric("Avg WPM", round(df['wpm'].mean()))
                h3.metric("Total Fillers Tracked", df['fillers'].sum())
                
                st.divider()
                st.markdown("**Speaking Pace (WPM) Over Time**")
                st.line_chart(df['wpm'], use_container_width=True)
                
                st.markdown("**Filler Word Count Over Time**")
                st.bar_chart(df['fillers'], use_container_width=True)
                
                # Raw Data
                with st.expander("View Raw Data"):
                    st.dataframe(df)
            else:
                st.info("No session history yet. Complete an analysis to see your progression!")

    except Exception as e:
        st.error("🚨 An unexpected error occurred.")
        st.code(str(e))
        logger.critical(f"Global Crash: {e}", exc_info=True)

if __name__ == "__main__":
    os.makedirs("temp_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    main()