import os
import sys
import streamlit as st
import platform
import shutil
import json
import logging
import ast
import time
import traceback
import pandas as pd
import plotly.express as px

# --- PATH INJECTION ---
# Ensures the project root is always at the top of sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- 1. ENVIRONMENT & LOGGING ---
from src.utils.diagnostics import get_logger, log_system_info

# Only log system info once per session to save performance
if 'sys_logged' not in st.session_state:
    log_system_info()
    st.session_state['sys_logged'] = True

logger = get_logger()

# --- 2. BACKEND & API ---
from src.api.client import APIClient
from src.ui.recorder import record_audio
from src.utils.file_manager import FileManager
from src.utils.pdf_generator import PDFGenerator
from src.utils.history import HistoryManager
from src.backend.personas import Personas

# --- CACHED API WRAPPERS ---
@st.cache_data(ttl=60) # Cache for 1 minute to prevent UI lag
def get_cached_models():
    """Prevents the UI from greying out by caching the network call to Ollama."""
    return APIClient.get_local_models()

# Ensure directories are ready
FileManager.initialize_directories()

try:
    import PIL
    from PIL import Image
except Exception as e:
    logger.debug(f"PIL import failed: {e}")

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
st.set_page_config(page_title="AI Interview Coach", page_icon="assets/Data-Drifters.png", layout="wide")

def main():
    try:
        # --- SIDEBAR: ALWAYS-ON MONITOR ---
        with st.sidebar:
            st.image("assets/Data-Drifters.png", width="stretch")
            st.header("🖥️ Live System Telemetry")
        
        st.title("🎙️ AI Interview Coach")
        
        compute_target = st.sidebar.radio("Compute Allocation", ["NVIDIA GPU", "CPU & RAM Core"], horizontal=True)
        
        # WRAP THE FRAGMENT IN THE SIDEBAR CONTEXT!
        with st.sidebar:
            live_hardware_monitor()

        st.sidebar.divider()
        
        # --- SIDEBAR: ENGINE CONFIGURATION ---
        st.sidebar.header("⚙️ Model Configuration")
        
        provider = st.sidebar.selectbox("Inference Provider", ["Local (Ollama)", "External API (Frontier Models)"])
        
        if provider == "Local (Ollama)":
            # Use cached model list to prevent UI lag
            downloaded_models = get_cached_models()
            selected_model = st.sidebar.selectbox("Local Model", downloaded_models)
            
            with st.sidebar.expander("⬇️ Download New Local Model"):
                common_ollama_models = [
                    "llama3.1:8b", "gemma2:9b", "mistral:7b", "phi3:latest", 
                    "llama3.1:70b", "gemma2:27b", "codellama:latest", "-- Other / Custom --"
                ]
                new_model_name = st.selectbox("Ollama Model Tag", common_ollama_models)
                
                if new_model_name == "-- Other / Custom --":
                    new_model_name = st.text_input("Enter Model Tag (e.g., dolphin-mixtral)")
                
                if st.button("Pull Model", width="stretch"):
                    progress_bar = st.progress(0, text="Initializing download...")
                    try:
                        response = APIClient.pull_model_stream(new_model_name)
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                status = data.get("status", "Downloading...")
                                if "completed" in data and "total" in data and data["total"] > 0:
                                    percent = data["completed"] / data["total"]
                                    progress_bar.progress(percent, text=f"{status} ({int(percent*100)}%)")
                                else:
                                    progress_bar.empty()
                                    st.sidebar.caption(f"Status: {status}")
                        
                        st.sidebar.success(f"✅ {new_model_name} ready!")
                        st.cache_data.clear() # Clear cache so new model shows up
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            
            api_key = None 
            
        else:
            api_service = st.sidebar.selectbox("API Target", ["OpenAI", "Anthropic", "Google Gemini"])
            
            if api_service == "OpenAI":
                external_models = ["gpt-5.4-nano", "o4-mini", "gpt-4.1-nano", "-- Other / Custom --"]
            elif api_service == "Anthropic":
                external_models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4-0", "-- Other / Custom --"]
            else: # Gemini
                external_models = ["gemini-3.1-flash-lite", "gemini-3-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite", "-- Other / Custom --"]
                
            selected_model = st.sidebar.selectbox("Model", external_models)
            if selected_model == "-- Other / Custom --":
                selected_model = st.sidebar.text_input("Custom Model String", placeholder="e.g., gpt-3.5-turbo")
            api_key = st.sidebar.text_input("Secret API Key", type="password")

        st.session_state['engine_config'] = {
            "provider": provider if provider == "Local (Ollama)" else api_service,
            "model": selected_model,
            "compute": compute_target,
            "api_key": api_key
        }

        # --- 🔌 CONNECTION STATUS & BUTTON ---
        st.sidebar.divider()
        st.sidebar.markdown("### 🔌 Connection Status")
        if 'connection_status' not in st.session_state:
            st.session_state['connection_status'] = {"success": False, "message": "⚪ Not tested yet."}
            
        if st.sidebar.button("Test Connection", width="stretch", type="primary"):
            with st.spinner("Pinging AI Engine..."):
                success, msg = APIClient.test_connection(st.session_state['engine_config'])
                st.session_state['connection_status'] = {"success": success, "message": msg}
                
        status = st.session_state['connection_status']
        if status["success"]:
            st.sidebar.success(status["message"])
        elif "⚪" in status["message"]:
            st.sidebar.info(status["message"])
        else:
            st.sidebar.error(status["message"])

        st.sidebar.divider()
        
        # --- SIDEBAR: RESET SESSION ---
        if st.sidebar.button("🔄 Start New Interview", use_container_width=True):
            # Keys to preserve (Engine Config, Hardware status, etc.)
            keys_to_keep = ['engine_config', 'connection_status', 'sys_logged']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()

        st.sidebar.divider()
        with st.sidebar.expander("🗑️ Danger Zone"):
            st.warning("This will permanently delete all session history and audio recordings.")
            if st.button("Delete All Data", width="stretch", type="primary"):
                files_deleted = FileManager.cleanup_all_data()
                HistoryManager.clear_history()
                st.success(f"Successfully cleared {files_deleted} files and history.")
                time.sleep(1)
                st.rerun()

        st.sidebar.divider()

        # --- MAIN INTERFACE ---
        tab_coach, tab_history = st.tabs(["🎯 Live Coach", "📈 Session History"])
        
        with tab_coach:
            if 'setup_step' not in st.session_state: st.session_state['setup_step'] = 1
            if 'rounds' not in st.session_state: st.session_state['rounds'] = []
            if 'custom_questions' not in st.session_state: st.session_state['custom_questions'] = []
            if 'round_info' not in st.session_state: st.session_state['round_info'] = {}
            
            with st.expander("🛠️ Interview Setup Wizard", expanded=(st.session_state['setup_step'] < 3)):
                st.markdown("### 1. Define Your Target Role")
                col_ind, col_role, col_sen = st.columns(3)
                industry = col_ind.text_input("Industry / Field", placeholder="e.g., Tech, Finance")
                job_title = col_role.text_input("Job Title", placeholder="e.g., Backend Developer")
                seniority = col_sen.selectbox("Seniority Level", ["Entry-Level", "Mid-Level", "Senior / Lead", "Executive"])
                
                if st.button("Generate Interview Rounds", disabled=not (industry and job_title)):
                    with st.spinner("🧠 API is structuring the interview process..."):
                        prompt = f"You are an expert tech recruiter. List 4 realistic interview rounds for a {seniority} {job_title} in the {industry} industry. Output ONLY a Python-style list of strings, nothing else."
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
                        list(Personas.PERSONA_PROMPTS.keys())
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
                            try:
                                import re
                                match = re.search(r"\[.*\]", q_response, re.DOTALL)
                                if match: questions = ast.literal_eval(match.group())
                                else: questions = [q.strip().lstrip('123456789. ') for q in q_response.split('\n') if q.strip()]
                            except:
                                questions = [q.strip() for q in q_response.split('\n') if q.strip()]
                            if not questions: questions = ["Could you tell me about your background?"]
                            questions.append("-- Custom Question --")
                            st.session_state['custom_questions'] = questions
                            st.session_state['setup_step'] = 3
                            st.rerun()

            if st.session_state['setup_step'] == 3:
                st.subheader("🎙️ Live Interview Simulator")
                info = st.session_state['round_info']
                st.info(f"⏱️ **Stage Context:** {info['meaning']} | **Interviewer:** {info['recommended_persona']}")

                if 'chat_history' not in st.session_state:
                    selected_persona_label = st.session_state.get('selected_persona_label', 'Standard HR')
                    base_prompt = Personas.PERSONA_PROMPTS.get(selected_persona_label, Personas.PERSONA_PROMPTS['Standard HR'])
                    selected_round = st.session_state.get('selected_round', 'General Interview')
                    sys_prompt = (
                        f"{base_prompt} You are conducting a {selected_round} interview for a {seniority} {job_title} role in the {industry} industry. "
                        f"Ask ONE question at a time. Keep questions concise. Follow-up on candidate's answers. Do not provide feedback yet."
                    )
                    st.session_state['system_prompt'] = sys_prompt
                    st.session_state['chat_history'] = []
                    st.session_state['aggregated_metrics'] = [] 
                    first_q = st.session_state['custom_questions'][0] if st.session_state['custom_questions'] else "Let's begin. Please introduce yourself."
                    st.session_state['chat_history'].append({"role": "assistant", "content": first_q})
                    speak_text(first_q)
                    st.divider()

                for msg in st.session_state['chat_history']:
                    avatar = "🤖" if msg["role"] == "assistant" else "👤"
                    with st.chat_message(msg["role"], avatar=avatar): st.write(msg["content"])

                st.divider()
                st.markdown("### Your Response")
                audio_path = record_audio()
                col_submit, col_end = st.columns(2)
                
                with col_submit:
                    if audio_path and st.button("🗣️ Submit Answer", type="primary", width="stretch"):
                        with st.spinner("Transcribing and processing..."):
                            compute_type = st.session_state['engine_config']['compute']
                            transcript, metrics, duration, error = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
                            if error: st.error(f"⚠️ {error}")
                            else:
                                st.session_state['aggregated_metrics'].append({"transcript": transcript, "metrics": metrics, "duration": duration})
                                st.session_state['chat_history'].append({"role": "user", "content": transcript})
                                with st.spinner("🧠 API is thinking..."):
                                    next_question = APIClient.generate_response(
                                        system_prompt=st.session_state['system_prompt'], 
                                        user_message=transcript, 
                                        chat_history=st.session_state['chat_history'][:-1], 
                                        engine_config=st.session_state['engine_config']
                                    )
                                    st.session_state['chat_history'].append({"role": "assistant", "content": next_question})
                                    speak_text(next_question)
                                    st.rerun()

                with col_end:
                    if len(st.session_state['chat_history']) > 1:
                        if st.button("🛑 End Interview & Analyze", width="stretch"):
                            if audio_path:
                                with st.spinner("Processing final answer..."):
                                    compute_type = st.session_state['engine_config']['compute']
                                    transcript, metrics, duration, error = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
                                    if not error:
                                        st.session_state['aggregated_metrics'].append({"transcript": transcript, "metrics": metrics, "duration": duration})
                                        st.session_state['chat_history'].append({"role": "user", "content": transcript})
                            st.session_state['interview_complete'] = True
                            st.rerun()

            if st.session_state.get('interview_complete', False):
                st.divider()
                st.header("📊 Final Interview Analysis")
                if 'final_feedback' not in st.session_state:
                    with st.spinner("🧠 API is compiling feedback..."):
                        total_wpm = total_fillers = total_duration = 0
                        valid_turns = len(st.session_state['aggregated_metrics'])
                        if valid_turns > 0:
                            for turn in st.session_state['aggregated_metrics']:
                                m = turn['metrics']
                                total_wpm += m['wpm']
                                total_fillers += m['filler_count']
                                total_duration += turn['duration']
                            avg_wpm = total_wpm / valid_turns
                        else: avg_wpm = 0
                        full_transcript = ""
                        for msg in st.session_state['chat_history']:
                            speaker = "Interviewer" if msg['role'] == "assistant" else "Candidate"
                            full_transcript += f"<b>{speaker}:</b> {msg['content']}<br><br>\n"
                        final_prompt = Personas.get_final_feedback_prompt(seniority, job_title, industry, full_transcript)
                        final_feedback = APIClient.generate_response(system_prompt=Personas.AI_COACH['system_prompt'], user_message=final_prompt, chat_history=[], engine_config=st.session_state['engine_config'])
                        try: HistoryManager.save_session(avg_wpm, total_fillers, "Multi-Turn", info['recommended_mode'])
                        except: pass
                        st.session_state['final_feedback'] = final_feedback
                        st.session_state['full_transcript'] = full_transcript
                        st.session_state['avg_wpm'] = avg_wpm
                        st.session_state['total_fillers'] = total_fillers
                        st.session_state['total_duration'] = total_duration

                tab_feedback, tab_metrics, tab_transcript = st.tabs(["🧠 AI Coach Feedback", "📈 Acoustic Metrics", "📝 Full Transcript"])
                with tab_feedback: st.markdown(st.session_state['final_feedback'])
                with tab_metrics:
                    col1, col2, col3 = st.columns(3)
                    mem_wpm = st.session_state['avg_wpm']
                    mem_fillers = st.session_state['total_fillers']
                    mem_duration = st.session_state['total_duration']
                    wpm_delta = "Ideal Pace" if 130 <= mem_wpm <= 160 else "Too Fast/Slow"
                    col1.metric("Average Pacing", f"{mem_wpm:.0f} WPM", delta=wpm_delta)
                    col2.metric("Total Filler Words", mem_fillers)
                    col3.metric("Total Speaking Time", f"{mem_duration:.1f}s")
                with tab_transcript: st.markdown(st.session_state['full_transcript'], unsafe_allow_html=True)
                
                pdf_path = os.path.join(FileManager.TEMP_DIR, f"report_{int(time.time())}.pdf")
                if PDFGenerator.generate_report(job_title, industry, {'wpm': mem_wpm, 'fillers': mem_fillers, 'duration': mem_duration}, st.session_state['final_feedback'], st.session_state['full_transcript'], pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(label="📄 Download PDF Report", data=f.read(), file_name=f"Report_{job_title}.pdf", mime="application/pdf")

        with tab_history:
            st.header("📈 Coaching Progress")
            history_data = HistoryManager.load_history()
            if not history_data: st.info("No history yet.")
            else:
                df = pd.DataFrame(history_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.line_chart(df.set_index('timestamp')[['wpm', 'fillers']])
                with st.expander("📝 View Logs"): st.dataframe(df.sort_values(by='timestamp', ascending=False))

    except Exception as e:
        st.error("🚨 A critical error occurred.")
        with st.expander("Details"): st.code(traceback.format_exc())
        logger.critical(f"Global UI Crash: {e}", exc_info=True)

if __name__ == "__main__":
    FileManager.initialize_directories()
    main()
