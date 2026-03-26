import os
import sys

# --- PATH INJECTION ---
# Ensures the project root is always at the top of sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Special Case: Sometimes Streamlit environment shadows 'src'
# or prefers importing submodules directly from 'src'
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from src.utils.diagnostics import get_logger, log_system_info
except ModuleNotFoundError:
    # Fallback for environments that treat 'src' as the project root
    from utils.diagnostics import get_logger, log_system_info

log_system_info()
logger = get_logger()

import platform
import shutil
import json
import logging
import ast
import time
import pandas as pd
import plotly.express as px
import streamlit as st

# --- 2. BACKEND & API ---
from src.api.client import APIClient
from src.ui.recorder import record_audio
from src.utils.file_manager import FileManager
from src.utils.pdf_generator import PDFGenerator
from src.utils.history import HistoryManager
from src.backend.personas import Personas

# Fallback imports if above fails (Streamlit pathing quirks)
try:
    from src.api.client import APIClient
except ModuleNotFoundError:
    from api.client import APIClient
    from ui.recorder import record_audio
    from utils.file_manager import FileManager
    from utils.pdf_generator import PDFGenerator
    from utils.history import HistoryManager
    from backend.personas import Personas

# Ensure directories are ready
FileManager.initialize_directories()

# --- LIVE HARDWARE FRAGMENT ---
@st.fragment(run_every=2)
def live_hardware_monitor():
    """Renders the sidebar telemetry dashboard."""
    hw_status = APIClient.get_hardware_status()
    if not hw_status:
        st.warning("⚠️ Cannot connect to API Telemetry.")
        return
        
    stats = hw_status.get("stats", {})
    has_nvidia = hw_status.get("has_nvidia", False) 
    gpu_detected = stats.get("gpu_detected", False) 
    gpu_name = stats.get("gpu_name")
    
    cpu_val = stats.get('cpu_percent') or 0
    ram_pct = stats.get('ram_percent') or 0
    ram_used = stats.get('ram_used_gb') or 0
    ram_total = stats.get('ram_total_gb') or 0
    
    st.progress(min(max(float(cpu_val) / 100.0, 0.0), 1.0), text=f"API CPU: {cpu_val}%")
    st.progress(min(max(float(ram_pct) / 100.0, 0.0), 1.0), text=f"API RAM: {ram_used}/{ram_total} GB")
    
    if gpu_detected:
        vram_pct = stats.get('vram_percent') or 0
        vram_used = stats.get('vram_used_gb') or 0
        vram_total = stats.get('vram_total_gb') or 0
        display_name = gpu_name if gpu_name else "NVIDIA GPU"
        
        status_suffix = " ⚠️ Torch Config Error" if not has_nvidia else ""
        text_label = f"API VRAM: {vram_used}/{vram_total} GB ({display_name}){status_suffix}"
        st.progress(min(max(float(vram_pct) / 100.0, 0.0), 1.0), text=text_label)
    else:
        if st.session_state.get('engine_config', {}).get('compute') == "NVIDIA GPU":
            st.caption("🔴 No NVIDIA GPU detected by API.")

# --- HELPERS ---
def speak_text(text):
    """Safely triggers the TTS engine in a background thread."""
    if not text: return
    
    import threading
    def run_tts():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            # Explicitly cleanup
            del engine
        except Exception as e:
            logger.debug(f"TTS Thread Error: {e}")

    # Start TTS in a background thread to avoid hanging Streamlit
    threading.Thread(target=run_tts, daemon=True).start()

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Interview Coach", page_icon="🎙️", layout="wide")

def main():
    try:
        st.title("🎙️ AI Interview Coach")
        
        # --- SIDEBAR: ALWAYS-ON MONITOR ---
        st.sidebar.header("🖥️ Live System Telemetry")
        
        compute_target = st.sidebar.radio("Compute Allocation", ["NVIDIA GPU", "CPU & RAM Core"], horizontal=True)
        
        # WRAP THE FRAGMENT IN THE SIDEBAR CONTEXT!
        # This prevents the progress bars from escaping into the main window
        with st.sidebar:
            live_hardware_monitor()

        st.sidebar.divider()
        
        # --- SIDEBAR: ENGINE CONFIGURATION ---
        st.sidebar.header("⚙️ Model Configuration")
        
        provider = st.sidebar.selectbox("Inference Provider", ["Local (Ollama)", "External API (Frontier Models)"])
        
        if provider == "Local (Ollama)":
            # 1. Dynamically fetch already downloaded models
            downloaded_models = APIClient.get_local_models()
            selected_model = st.sidebar.selectbox("Local Model", downloaded_models)
            
            with st.sidebar.expander("⬇️ Download New Local Model"):
                new_model_name = st.text_input("Ollama Model Tag (e.g., gemma2:9b)")
                
                if st.button("Pull Model", use_container_width=True):
                    progress_bar = st.progress(0, text="Initializing download...")
                    try:
                        # 2. Start the stream
                        response = APIClient.pull_model_stream(new_model_name)
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                status = data.get("status", "Downloading...")
                                
                                # 3. Calculate percentage if bytes are provided
                                if "completed" in data and "total" in data and data["total"] > 0:
                                    percent = data["completed"] / data["total"]
                                    progress_bar.progress(percent, text=f"{status} ({int(percent*100)}%)")
                                else:
                                    progress_bar.empty()
                                    st.sidebar.caption(f"Status: {status}")
                        
                        st.sidebar.success(f"✅ {new_model_name} ready!")
                        st.rerun() # Refresh to show new model in dropdown
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            
            api_key = None 
            
        else:
            api_service = st.sidebar.selectbox("API Target", ["OpenAI", "Anthropic", "Google Gemini"])
            selected_model = st.sidebar.text_input("Model String", placeholder="e.g., gpt-4o, claude-3-5-sonnet")
            api_key = st.sidebar.text_input("Secret API Key", type="password")
            
        # Save these settings to the session state
        st.session_state['engine_config'] = {
            "provider": provider if provider == "Local (Ollama)" else api_service,
            "model": selected_model,
            "compute": compute_target,
            "api_key": api_key
        }

        # --- 🔌 CONNECTION STATUS & BUTTON ---
        st.sidebar.divider()
        st.sidebar.markdown("### 🔌 Connection Status")
        
        # Initialize state if it doesn't exist
        if 'connection_status' not in st.session_state:
            st.session_state['connection_status'] = {"success": False, "message": "⚪ Not tested yet."}
            
        if st.sidebar.button("Test Connection", use_container_width=True, type="primary"):
            with st.spinner("Pinging AI Engine..."):
                success, msg = APIClient.test_connection(st.session_state['engine_config'])
                st.session_state['connection_status'] = {"success": success, "message": msg}
                
        # Display the result
        status = st.session_state['connection_status']
        if status["success"]:
            st.sidebar.success(status["message"])
        elif "⚪" in status["message"]:
            st.sidebar.info(status["message"])
        else:
            st.sidebar.error(status["message"])

        st.sidebar.divider()
        
        # --- SIDEBAR: CLEANUP ---
        with st.sidebar.expander("🗑️ Danger Zone"):
            st.warning("This will permanently delete all session history and audio recordings.")
            if st.button("Delete All Data", use_container_width=True, type="primary"):
                files_deleted = FileManager.cleanup_all_data()
                HistoryManager.clear_history()
                st.success(f"Successfully cleared {files_deleted} files and history.")
                time.sleep(1)
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
                            engine_config=st.session_state['engine_config']
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
                    
                    st.divider()
                    st.markdown("### 3. Choose Interviewer Persona")
                    selected_persona_label = st.selectbox(
                        "Select the Interviewer Style:", 
                        list(Personas.PERSONA_PROMPTS.keys()),
                        help="This determines the AI's tone, aggression, and focus area."
                    )
                    
                    if st.button("Generate Custom Questions", type="primary"):
                        with st.spinner(f"🧠 API is writing questions for the {selected_round}..."):
                            
                            st.session_state['selected_persona_label'] = selected_persona_label
                            st.session_state['selected_round'] = selected_round
                            
                            round_type = selected_round.split(" ")[0]
                            persona_config = Personas.get_interviewer_by_type(round_type, seniority)

                            st.session_state['round_info'] = {
                                "meaning": persona_config['meaning'], 
                                "recommended_mode": persona_config['recommended_mode'],
                                "recommended_persona": persona_config['persona']
                            }
                            
                            q_prompt = f"Generate 3 highly specific interview questions for a {seniority} {job_title} during the '{selected_round}' round. Output ONLY a Python-style list of strings."
                            q_response = APIClient.generate_response(
                                system_prompt="You are an expert interviewer. You output ONLY a Python-style list of strings.", 
                                user_message=q_prompt, 
                                chat_history=[], 
                                engine_config=st.session_state['engine_config']
                            )
                            
                            # --- ROBUST PARSING ---
                            try:
                                # Try to find a list [ ... ] within the response
                                import re
                                match = re.search(r"\[.*\]", q_response, re.DOTALL)
                                if match:
                                    questions = ast.literal_eval(match.group())
                                else:
                                    # Fallback: Split by lines and clean
                                    questions = [q.strip().lstrip('123456789. ') for q in q_response.split('\n') if q.strip()]
                            except:
                                questions = [q.strip() for q in q_response.split('\n') if q.strip()]
                            
                            # Ensure we actually got questions
                            if not questions:
                                questions = ["Could you tell me about your background?", "Why are you interested in this role?"]
                                
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
                    # Get the base prompt from our new mapping
                    selected_persona_label = st.session_state.get('selected_persona_label', 'Standard HR')
                    base_prompt = Personas.PERSONA_PROMPTS.get(selected_persona_label, Personas.PERSONA_PROMPTS['Standard HR'])
                    
                    selected_round = st.session_state.get('selected_round', 'General Interview')
                    
                    sys_prompt = (
                        f"{base_prompt} "
                        f"You are conducting a {selected_round} interview for a {seniority} {job_title} role in the {industry} industry. "
                        f"Ask ONE question at a time. Keep your questions concise (1-2 sentences). "
                        f"Base your follow-ups strictly on the candidate's previous answer. "
                        f"Do not break character. Do not provide feedback yet."
                    )
                    
                    st.session_state['system_prompt'] = sys_prompt
                    st.session_state['chat_history'] = []
                    st.session_state['aggregated_metrics'] = [] 
                    
                    # Ensure first_q is a real string
                    first_q = st.session_state['custom_questions'][0] if st.session_state['custom_questions'] else "Let's begin. Please introduce yourself."
                    
                    st.session_state['chat_history'].append({"role": "assistant", "content": first_q})
                    
                    # Call the safe speaker
                    speak_text(first_q)

                    st.divider()

                for msg in st.session_state['chat_history']:
                    avatar = "🤖" if msg["role"] == "assistant" else "👤"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])

                st.divider()
                st.markdown("### Your Response")
                
                # Solely relying on the live microphone
                audio_path = record_audio()

                col_submit, col_end = st.columns(2)
                
                with col_submit:
                    # Submit button only activates if a recording exists
                    if audio_path and st.button("🗣️ Submit Answer", type="primary", use_container_width=True):
                        with st.spinner("Transcribing and processing..."):
                            # We pass 'NVIDIA GPU' or 'CPU' directly from your sidebar config
                            compute_type = st.session_state['engine_config']['compute']
                            
                            transcript, metrics, duration, error = APIClient.process_audio(
                                audio_path, 
                                info['recommended_mode'], 
                                compute_type
                            )
                            
                            if error:
                                st.error(f"⚠️ {error}")
                            else:
                                # Update history and state
                                st.session_state['aggregated_metrics'].append({
                                    "transcript": transcript,
                                    "metrics": metrics,
                                    "duration": duration
                                })
                                st.session_state['chat_history'].append({"role": "user", "content": transcript})
                                
                                # Generate the next AI question
                                with st.spinner("🧠 API is thinking..."):
                                    next_question = APIClient.generate_response(
                                        system_prompt=st.session_state['system_prompt'], 
                                        user_message=transcript, 
                                        chat_history=st.session_state['chat_history'][:-1], 
                                        engine_config=st.session_state['engine_config']
                                    )
                                    
                                    st.session_state['chat_history'].append({"role": "assistant", "content": next_question})
                                    
                                    # Trigger the TTS to speak the question
                                    speak_text(next_question)
                                    
                                    st.rerun()

                with col_end:
                    if len(st.session_state['chat_history']) > 1:
                        if st.button("🛑 End Interview & Analyze", use_container_width=True):
                            # Process the final recording if one exists before ending
                            if audio_path:
                                with st.spinner("Processing final answer..."):
                                    compute_type = st.session_state['engine_config']['compute']
                                    transcript, metrics, duration, error = APIClient.process_audio(
                                        audio_path, 
                                        info['recommended_mode'], 
                                        compute_type
                                    )
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
                            
                        # 2. Use Personas for prompt generation
                        final_prompt = Personas.get_final_feedback_prompt(
                            seniority, job_title, industry, full_transcript
                        )
                        
                        final_feedback = APIClient.generate_response(
                            system_prompt=Personas.AI_COACH['system_prompt'], 
                            user_message=final_prompt, 
                            chat_history=[], 
                            engine_config=st.session_state['engine_config']
                        )
                        
                        try:
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
                    st.markdown(st.session_state['full_transcript'], unsafe_allow_html=True)
                    
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
            
            # --- 1. LOAD & SORT HISTORY ---
            def get_sorted_history():
                """Loads and sorts history by date (oldest to newest for plotting)."""
                try:
                    data = HistoryManager.load_history()
                    if not data: return []
                    # Sort by timestamp string (ISO-like format: YYYY-MM-DD HH:MM)
                    return sorted(data, key=lambda x: x['timestamp'])
                except Exception:
                    return []

            history_data = get_sorted_history()

            if not history_data:
                st.info("No session history found. Complete your first practice interview to see your progress here!")
            else:
                df = pd.DataFrame(history_data)
                df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure it's datetime for plotting

                # --- 2. SESSION COMPARISON ANALYTICS (DELTAS) ---
                st.subheader("📊 Session Comparison")
                
                if len(history_data) >= 2:
                    current_session = history_data[-1]
                    previous_session = history_data[-2]
                    
                    def calc_delta(curr, prev):
                        if prev == 0: return 0
                        return ((curr - prev) / prev) * 100

                    # Calculate Deltas
                    wpm_delta = calc_delta(current_session['wpm'], previous_session['wpm'])
                    filler_delta = calc_delta(current_session['fillers'], previous_session['fillers'])
                    
                    # Layout Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # WPM Metric (Increasing is usually good)
                    col1.metric(
                        label="Avg WPM Change",
                        value=f"{current_session['wpm']:.1f}",
                        delta=f"{wpm_delta:+.1f}% vs last",
                        delta_color="normal" 
                    )
                    
                    # Fillers Metric (Decreasing is GOOD - use 'inverse' for green on negative)
                    col2.metric(
                        label="Filler Word Change",
                        value=current_session['fillers'],
                        delta=f"{filler_delta:+.1f}% vs last",
                        delta_color="inverse" 
                    )
                    
                    col3.metric("Current Mode", current_session['mode'])
                    st.caption("💡 *Note: A green delta for Filler Words indicates a reduction, which is an improvement!*")
                else:
                    st.info("💡 Practice at least twice to see session-to-session comparison analytics and deltas.")

                st.divider()

                # --- 3. TREND VISUALIZATION (MULTI-LINE CHART) ---
                st.subheader("📈 Performance Trends")
                
                if len(history_data) >= 2:
                    # Melt the dataframe for multi-line plotting if needed, 
                    # but Plotly Express can handle multiple Y columns directly.
                    fig = px.line(
                        df, 
                        x='timestamp', 
                        y=['wpm', 'fillers'],
                        markers=True,
                        title="WPM vs. Filler Words Over Time",
                        labels={"value": "Count / Rate", "timestamp": "Session Date", "variable": "Metric"},
                        color_discrete_map={"wpm": "#00CC96", "fillers": "#EF553B"}
                    )
                    
                    fig.update_layout(
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Simple single chart for first session
                    st.caption("Trend charts will expand as you complete more sessions.")
                    fig_simple = px.scatter(df, x='timestamp', y='wpm', title="Initial Progress Point")
                    st.plotly_chart(fig_simple, use_container_width=True)

                st.divider()

                # --- 4. RAW LOGS ---
                with st.expander("📝 View Detailed Session Logs"):
                    st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("🚨 A critical application error occurred.")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        logger.critical(f"Global UI Crash: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        FileManager.initialize_directories()
        main()
    except Exception as e:
        # Emergency fallback if main fails before Streamlit initializes
        print(f"CRITICAL SYSTEM FAILURE: {e}")