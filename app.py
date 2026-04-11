import os
import sys
import streamlit as st
import json
import ast
import time
import traceback
import base64
import re
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- PATH INJECTION ---
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path: sys.path.insert(0, root_dir)
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path: sys.path.insert(0, src_dir)

# --- INTERNAL IMPORTS ---
from src.utils.diagnostics import get_logger, log_system_info
from src.api.client import APIClient
from src.ui.recorder import record_audio
from src.utils.file_manager import FileManager
from src.utils.history import HistoryManager
from src.backend.hardware import HardwareInfo
from src.backend.personas import Personas
from src.utils.text_processor import clean_llm_text, parse_file
from src.ui.dashboard import render_final_analysis, render_history_dashboard

# --- INITIALIZATION ---
if 'sys_logged' not in st.session_state:
    log_system_info()
    st.session_state['sys_logged'] = True

logger = get_logger()
FileManager.initialize_directories()

# --- CACHED WRAPPERS ---
@st.cache_data(ttl=300)
def get_cached_models():
    return APIClient.get_local_models()

# --- UI COMPONENTS ---
@st.fragment(run_every=1)
def unified_status_monitor(config):
    """Renders live telemetry and connection indicators."""
    hw_status = APIClient.get_hardware_status()
    st.subheader("🔌 Connection Status")
    
    if hw_status:
        st.success("🟢 Backend: Online")
        if config:
            success, msg = APIClient.test_connection(config)
            if success: st.success("🟢 AI Engine: Ready")
            else: st.error(msg)
        
        st.divider()
        st.subheader("🖥️ Resource Usage")
        stats = hw_status.get("stats", {})
        c_v, r_p = stats.get('cpu_percent', 0), stats.get('ram_percent', 0)
        r_u, r_t = stats.get('ram_used_gb', 0), stats.get('ram_total_gb', 0)
        st.progress(min(max(float(c_v) / 100.0, 0.0), 1.0), text=f"CPU: {c_v}%")
        st.progress(min(max(float(r_p) / 100.0, 0.0), 1.0), text=f"RAM: {r_u}/{r_t} GB")
        if stats.get("gpu_detected"):
            v_p = stats.get('vram_percent') or 0
            v_u = stats.get('vram_used_gb') or 0
            v_t = stats.get('vram_total_gb') or 0
            st.progress(min(max(float(v_p) / 100.0, 0.0), 1.0), text=f"GPU VRAM: {v_u}/{v_t} GB")
    else:
        st.error("🔴 Backend: Offline")

def trigger_voice(text):
    """Fetches audio and sets play state."""
    if not text: return
    clean_text = clean_llm_text(text)
    voice = st.session_state.get('selected_voice', "en-US-GuyNeural")
    audio_bytes = APIClient.generate_speech(clean_text, voice=voice)
    if audio_bytes:
        st.session_state['play_now_bytes'] = audio_bytes
        st.session_state['audio_nonce'] = st.session_state.get('audio_nonce', 0) + 1
        return True
    return False

@st.fragment
def isolated_recorder_flow(info):
    """Isolates recorder widget for UI stability."""
    st.divider()
    if 'rec_nonce' not in st.session_state: st.session_state['rec_nonce'] = 0
    audio_path = record_audio(key=f"recorder_{st.session_state['rec_nonce']}")
    c_sub, c_end = st.columns(2)
    
    compute_type = st.session_state.get('engine_config', {}).get('compute', 'CPU & RAM Core')
    
    if audio_path and c_sub.button("🗣️ Submit Answer", type="primary", width='stretch'):
        with st.spinner("Processing..."):
            t, m, d, e = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
            if not e:
                st.session_state['aggregated_metrics'].append({"transcript": t, "metrics": m, "duration": d})
                st.session_state['chat_history'].append({"role": "user", "content": t})
                with st.spinner("Thinking..."):
                    nxt = APIClient.generate_response(st.session_state['sys_p'], t, st.session_state['chat_history'][:-1], st.session_state['engine_config'], resume_context=st.session_state.get('resume_text', ""), job_context=st.session_state.get('job_desc_text', ""))
                    trigger_voice(nxt)
                    st.session_state['chat_history'].append({"role": "assistant", "content": clean_llm_text(nxt)})
                    st.session_state['rec_nonce'] += 1
                    st.rerun()

    if c_end.button("🛑 End Interview & Analyze", width='stretch'):
        if audio_path:
            with st.spinner("Finalizing..."):
                t, m, d, e = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
                if not e:
                    st.session_state['aggregated_metrics'].append({"transcript": t, "metrics": m, "duration": d})
                    st.session_state['chat_history'].append({"role": "user", "content": t})
        st.session_state['interview_complete'] = True
        st.rerun()

# --- MAIN APP ---
st.set_page_config(page_title="AI Interview Coach", page_icon="assets/Data-Drifters.png", layout="wide")

def main():
    try:
        # --- STARTUP AUTO-DETECTION ---
        if 'default_provider' not in st.session_state:
            try:
                res = requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
                if res.status_code == 200: st.session_state['default_provider'] = "Ollama (Local)"
                else: st.session_state['default_provider'] = "Google Gemini"
            except: st.session_state['default_provider'] = "Google Gemini"

        # --- PERSISTENT KEY LOADING ---
        if 'saved_keys' not in st.session_state:
            st.session_state['saved_keys'] = FileManager.load_saved_keys()

        # --- SIDEBAR ---
        with st.sidebar:
            st.image("assets/Data-Drifters.png", width="stretch")
            
            st.header("⚙️ Configuration")
            
            # 1. Compute
            hw = HardwareInfo()
            rec_mode, rec_text = hw.get_recommendation()
            c_options = ["NVIDIA GPU", "CPU & RAM Core"]
            if hw.is_apple_silicon: c_options = ["Apple Silicon", "CPU & RAM Core"]
            compute_target = st.radio("Compute Allocation", c_options, horizontal=True)
            with st.expander("💡 Hardware Helper"):
                st.info(rec_text)
                st.caption(f"**Detected:** {hw.cpu_info}")
            
            # 2. Voice
            v_opt = st.radio("Coach Voice", ["Male", "Female"], horizontal=True)
            st.session_state['selected_voice'] = "en-US-GuyNeural" if v_opt == "Male" else "en-US-AvaNeural"

            # 3. Provider
            providers = ["Ollama (Local)", "OpenAI", "Google Gemini", "Anthropic"]
            default_idx = providers.index(st.session_state.get('default_provider', "Google Gemini"))
            provider = st.selectbox("Inference Provider", providers, index=default_idx)
            
            if provider == "Ollama (Local)":
                models = get_cached_models()
                selected_model = st.selectbox("Local Model", models)
                with st.expander("⬇️ Download Model"):
                    tag = st.selectbox("Tag", ["llama3.1:8b", "gemma2:9b", "mistral:7b", "-- Other --"])
                    if tag == "-- Other --": tag = st.text_input("Custom Tag")
                    if st.button("Pull Model", use_container_width=True):
                        p = st.progress(0, "Starting...")
                        res = APIClient.pull_model_stream(tag)
                        for line in res.iter_lines():
                            if line:
                                d = json.loads(line.decode('utf-8'))
                                if "completed" in d and "total" in d and d["total"] > 0: p.progress(d["completed"]/d["total"], f"Downloading {tag}...")
                        st.success("Ready!"); st.cache_data.clear(); st.rerun()
                api_key = None
            else:
                if provider == "OpenAI": m_list = ["gpt-5.4-nano", "gpt-5-nano", "o4-mini", "gpt-4o-mini"]
                elif provider == "Anthropic": m_list = ["claude-sonnet-4-6","claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-0"]
                else: m_list = ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite"]
                
                selected_model = st.selectbox("Model", m_list + ["-- Other --"])
                if selected_model == "-- Other --": selected_model = st.text_input("Custom Model")
                
                st.caption(f"ℹ️ [Get API Key](https://aistudio.google.com/app/api-keys)")
                api_key = st.text_input("Secret API Key", value=st.session_state['saved_keys'].get(provider, ""), type="password")
                if api_key != st.session_state['saved_keys'].get(provider):
                    st.session_state['saved_keys'][provider] = api_key
                    FileManager.save_keys(st.session_state['saved_keys'])

            # --- ENGINE CONFIG ---
            st.session_state['engine_config'] = {"provider": provider, "model": selected_model, "compute": compute_target, "api_key": api_key}

            # --- 4. STATUS (REACTIVE) ---
            st.divider()
            unified_status_monitor(st.session_state['engine_config'])

            st.divider()
            if st.button("🔄 Start New Interview", use_container_width=True):
                keep = ['engine_config', 'sys_logged', 'selected_voice', 'saved_keys', 'default_provider', 'wipe_nonce']
                # Increment wipe_nonce to force-reset input fields
                st.session_state['wipe_nonce'] = st.session_state.get('wipe_nonce', 0) + 1
                for k in list(st.session_state.keys()):
                    if k not in keep: del st.session_state[k]
                st.rerun()

            with st.expander("🗑️ Danger Zone"):
                if st.button("Delete All Data", type="primary", use_container_width=True):
                    # 1. Clear local files
                    FileManager.cleanup_all_data(); HistoryManager.clear_history()
                    FileManager.safe_delete_file(os.path.join(FileManager.TEMP_DIR, "vault.json"))
                    
                    # 2. Increment reset counter
                    new_nonce = st.session_state.get('wipe_nonce', 0) + 1
                    
                    # 3. Wipe everything EXCEPT the reset counter
                    for k in list(st.session_state.keys()): 
                        del st.session_state[k]
                    
                    st.session_state['wipe_nonce'] = new_nonce
                    st.rerun()

        st.title("🎙️ AI Interview Coach")

        if st.session_state.get('play_now_bytes'):
            b64 = base64.b64encode(st.session_state['play_now_bytes']).decode()
            n = st.session_state.get('audio_nonce', 0)
            st.components.v1.html(f'<audio id="a_{n}" autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio><script>document.getElementById("a_{n}").play();</script>', height=1)
            st.session_state['play_now_bytes'] = None

        tab_coach, tab_history = st.tabs(["🎯 Live Coach", "📈 Session History"])
        
        with tab_coach:
            if 'setup_step' not in st.session_state: st.session_state['setup_step'] = 1
            if 'rounds' not in st.session_state: st.session_state['rounds'] = []
            if 'aggregated_metrics' not in st.session_state: st.session_state['aggregated_metrics'] = []
            if 'wipe_nonce' not in st.session_state: st.session_state['wipe_nonce'] = 0
            
            w_n = st.session_state['wipe_nonce']

            with st.expander("🛠️ Setup Wizard", expanded=(st.session_state['setup_step'] < 3)):
                st.markdown("### 1. Define Target Role")
                c1, c2, c3 = st.columns(3)
                # Adding keys to widgets allows us to clear them via session_state deletion
                ind = c1.text_input("Industry", placeholder="e.g., Tech", key=f"ind_in_{w_n}")
                role = c2.text_input("Job Title", placeholder="e.g., Backend Developer", key=f"role_in_{w_n}")
                sen = c3.selectbox("Seniority", ["Entry-Level", "Mid-Level", "Senior / Lead", "Executive"], key=f"sen_in_{w_n}")
                
                st.divider()
                st.markdown("### 2. Contextual Data (Optional)")
                c_res, c_jd = st.columns(2)
                res_f = c_res.file_uploader("Upload Resume", type=["pdf", "txt", "docx"], key=f"res_up_{w_n}")
                jd_f = c_jd.file_uploader("Upload Job Description", type=["pdf", "txt", "docx"], key=f"jd_up_{w_n}")
                
                ALLOWED_EXT = [".pdf", ".txt", ".docx"]
                MAX_SIZE_MB = 200
                
                # Use standard logic but check widget keys
                if res_f:
                    if res_f.size > MAX_SIZE_MB * 1024 * 1024:
                        st.error(f"❌ File too large: {res_f.name}")
                        st.session_state['resume_text'] = ""
                    else:
                        res_text = parse_file(res_f)
                        if res_text: st.session_state['resume_text'] = res_text
                        else: st.session_state['resume_text'] = None
                else: st.session_state['resume_text'] = ""

                if jd_f:
                    if jd_f.size > MAX_SIZE_MB * 1024 * 1024:
                        st.error(f"❌ File too large: {jd_f.name}")
                        st.session_state['job_desc_text'] = ""
                    else:
                        jd_text = parse_file(jd_f)
                        if jd_text: st.session_state['job_desc_text'] = jd_text
                        else: st.session_state['job_desc_text'] = None
                else: st.session_state['job_desc_text'] = ""

                if st.button("Generate Interview Rounds", disabled=not (ind and role)):
                    with st.spinner("Structuring..."):
                        p = (f"List 4 unique interview rounds for a {sen} {role} in the {ind} industry. "
                             f"Output ONLY a Python list of strings. Do NOT output anything else.")
                        resp = APIClient.generate_response("You are an expert recruiter. Return ONLY a Python list.", p, [], st.session_state['engine_config'])
                        try:
                            import re; m = re.search(r"\[.*\]", resp, re.DOTALL)
                            rounds = ast.literal_eval(m.group()) if m else [resp]
                            st.session_state['rounds'] = [str(r).strip() for r in rounds if len(str(r)) < 150]
                        except: st.session_state['rounds'] = ["1. Initial Screen", "2. Technical Round", "3. Culture Fit", "4. Final Manager"]
                        st.session_state['setup_step'] = 2; st.rerun()

                if st.session_state['setup_step'] >= 2:
                    st.divider()
                    st.markdown("### 3. Select Stage & Persona")
                    sel_r = st.selectbox("Target Interview Round:", st.session_state['rounds'], key=f"round_sel_{w_n}")
                    sel_p = st.selectbox("Interviewer Style:", list(Personas.PERSONA_PROMPTS.keys()), key=f"persona_sel_{w_n}")
                    if st.button("Generate Questions", type="primary"):
                        if not sel_r: st.error("Please generate interview rounds in Step 1 first!")
                        else:
                            with st.spinner("Writing..."):
                                st.session_state.update({'selected_persona_label': sel_p, 'selected_round': sel_r, 'seniority': sen, 'job_title': role, 'industry': ind})
                                persona_cfg = Personas.get_interviewer_by_type(sel_r.split(" ")[0], sen)
                                st.session_state['round_info'] = {"meaning": persona_cfg['meaning'], "persona": persona_cfg['persona'], "recommended_mode": "Balanced"}
                                q_resp = APIClient.generate_questions(sen, role, ind, sel_r, st.session_state['engine_config'], st.session_state['resume_text'], st.session_state['job_desc_text'])
                                raw_qs = [q.strip() for q in q_resp.split('\n') if len(q.strip()) > 10]
                                st.session_state['custom_questions'] = [clean_llm_text(q) for q in raw_qs]
                                st.session_state['setup_step'] = 3; st.rerun()

            if st.session_state['setup_step'] == 3 and not st.session_state.get('interview_complete'):
                st.subheader("🎙️ Interview Simulator")
                info = st.session_state['round_info']; st.info(f"⏱️ **Stage:** {info['meaning']} | **Interviewer:** {info['persona']}")
                if 'chat_history' not in st.session_state:
                    st.session_state['sys_p'] = Personas.get_interview_sys_prompt(st.session_state['selected_persona_label'], st.session_state['selected_round'], st.session_state['seniority'], st.session_state['job_title'], st.session_state['industry'], st.session_state['resume_text'], st.session_state['job_desc_text'])
                    st.session_state['chat_history'] = []
                    with st.spinner("🎙️ Coach is entering the room..."):
                        first_q = APIClient.generate_response(st.session_state['sys_p'], "Start the interview. Greet me and ask ONLY your first question.", [], st.session_state['engine_config'], resume_context=st.session_state.get('resume_text', ""), job_context=st.session_state.get('job_desc_text', ""))
                        first_q = clean_llm_text(first_q); trigger_voice(first_q)
                        st.session_state['chat_history'].append({"role": "assistant", "content": first_q}); st.rerun()

                for idx, msg in enumerate(st.session_state['chat_history']):
                    with st.chat_message(msg["role"], avatar=("🤖" if msg["role"] == "assistant" else "👤")):
                        st.write(msg["content"])
                        if msg["role"] == "assistant":
                            if st.button("🔊 Replay", key=f"replay_{idx}"): trigger_voice(msg["content"]); st.rerun()
                isolated_recorder_flow(info)

            if st.session_state.get('interview_complete'):
                if 'final_feedback' not in st.session_state:
                    with st.spinner("Analyzing..."):
                        v = len(st.session_state['aggregated_metrics']); t_w = sum([m['metrics']['wpm'] for m in st.session_state['aggregated_metrics']])
                        t_f = sum([m['metrics']['filler_count'] for m in st.session_state['aggregated_metrics']]); t_d = sum([m['duration'] for m in st.session_state['aggregated_metrics']])
                        avg_w = t_w / v if v > 0 else 0; full_t = ""
                        for msg in st.session_state['chat_history']: full_t += f"<b>{'Interviewer' if msg['role'] == 'assistant' else 'Candidate'}:</b> {msg['content']}<br><br>\n"
                        f_p = Personas.get_final_feedback_prompt(st.session_state['seniority'], st.session_state['job_title'], st.session_state['industry'], full_t)
                        f_f = APIClient.generate_response(Personas.AI_COACH['system_prompt'], f_p, [], st.session_state['engine_config'], resume_context=st.session_state['resume_text'])
                        HistoryManager.save_session(avg_w, t_f, "Analysis", "Multi-Turn")
                        st.session_state.update({'final_feedback': f_f, 'full_transcript': full_t, 'avg_wpm': avg_w, 'total_fillers': t_f, 'total_duration': t_d})
                render_final_analysis(st.session_state)

        with tab_history:
            render_history_dashboard()

    except Exception as e:
        st.error(f"🚨 Error: {e}"); st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
