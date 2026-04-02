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
import base64
import re

# --- PATH INJECTION ---
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path: sys.path.insert(0, root_dir)
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path: sys.path.insert(0, src_dir)

# --- 1. ENVIRONMENT & LOGGING ---
from src.utils.diagnostics import get_logger, log_system_info

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
@st.cache_data(ttl=60)
def get_cached_models():
    return APIClient.get_local_models()

@st.cache_data(ttl=5) # Prevent flicker by caching probe results for 5s
def cached_probe(config):
    return APIClient.test_connection(config)

# Ensure directories are ready
FileManager.initialize_directories()

# --- LIVE STATUS FRAGMENT (Isolated to reduce grey-out) ---
@st.fragment(run_every=3) # Slower heartbeat to reduce visual distraction
def unified_status_monitor():
    """Renders all live telemetry and connection indicators inside a container."""
    # We use a container to visually group these and minimize sidebar layout shifts
    with st.container():
        hw_status = APIClient.get_hardware_status()
        
        # 1. Backend Status
        st.subheader("🔌 Connection Status")
        if hw_status:
            st.success("🟢 Backend: Online")
        else:
            st.error("🔴 Backend: Offline")
            st.warning("⚠️ Telemetry Offline")
            return

        # 2. AI Engine Status
        if 'engine_config' in st.session_state:
            # Use cached probe to avoid network delay on every single tick
            success, msg = cached_probe(st.session_state['engine_config'])
            if success: st.success("🟢 AI Engine: Ready")
            else: st.error(f"🔴 AI Engine: {msg.split('.')[0]}")

        # 3. Hardware Telemetry
        st.divider()
        st.subheader("🖥️ Resource Usage")
        stats = hw_status.get("stats", {})
        c_val, r_pct = stats.get('cpu_percent', 0), stats.get('ram_percent', 0)
        r_u, r_t = stats.get('ram_used_gb', 0), stats.get('ram_total_gb', 0)
        
        st.progress(min(max(float(c_val) / 100.0, 0.0), 1.0), text=f"API CPU: {c_val}%")
        st.progress(min(max(float(r_pct) / 100.0, 0.0), 1.0), text=f"API RAM: {r_u}/{r_t} GB")
        
        if stats.get("gpu_detected"):
            v_pct = stats.get('vram_percent', 0)
            v_u, v_t = stats.get('vram_used_gb', 0), stats.get('vram_total_gb', 0)
            st.progress(min(max(float(v_pct) / 100.0, 0.0), 1.0), text=f"API VRAM: {v_u}/{v_t} GB")

# --- HELPERS ---
def clean_llm_text(text):
    raw = str(text).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0: raw = str(parsed[0])
    except: raw = raw.strip('[]"\' ')
    raw = re.sub(r'^(Question\s*\d+:|\d+\.)\s*', '', raw, flags=re.IGNORECASE)
    return raw.strip()

def parse_file(uploaded_file):
    if not uploaded_file: return None
    try:
        if uploaded_file.type == "text/plain": return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            import pypdf
            return " ".join([p.extract_text() for p in pypdf.PdfReader(uploaded_file).pages])
    except: return "Error parsing file."
    return None

def trigger_voice(text):
    if not text: return
    clean_text = clean_llm_text(text)
    voice = st.session_state.get('selected_voice', "en-US-GuyNeural")
    audio_bytes = APIClient.generate_speech(clean_text, voice=voice)
    if audio_bytes:
        st.session_state['play_now_bytes'] = audio_bytes
        st.session_state['audio_nonce'] = st.session_state.get('audio_nonce', 0) + 1
        return True
    return False

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Interview Coach", page_icon="assets/Data-Drifters.png", layout="wide")

def main():
    try:
        # --- SIDEBAR: LIVE MONITORING ---
        with st.sidebar:
            st.image("assets/Data-Drifters.png", width="stretch")
            
            # --- Dedicated Fragment Container ---
            # This helps isolate the visual refresh to just this section
            with st.container():
                unified_status_monitor()
            
            st.divider()
            st.header("⚙️ Configuration")
            v_opt = st.radio("Coach Voice", ["Male", "Female"], horizontal=True)
            voice_map = {"Male": "en-US-GuyNeural", "Female": "en-US-AvaNeural"}
            st.session_state['selected_voice'] = voice_map[v_opt]

            provider = st.selectbox("Inference Provider", ["Local (Ollama)", "External API"])
            if provider == "Local (Ollama)":
                models = get_cached_models()
                selected_model = st.selectbox("Local Model", models)
                with st.expander("⬇️ Download New Model"):
                    tag = st.selectbox("Tag", ["llama3.1:8b", "gemma2:9b", "mistral:7b", "-- Other --"])
                    if tag == "-- Other --": tag = st.text_input("Custom Tag")
                    if st.button("Pull Model", use_container_width=True):
                        p = st.progress(0, "Starting...")
                        res = APIClient.pull_model_stream(tag)
                        for line in res.iter_lines():
                            if line:
                                d = json.loads(line.decode('utf-8'))
                                if "completed" in d and "total" in d and d["total"] > 0:
                                    p.progress(d["completed"]/d["total"], f"Downloading {tag}...")
                        st.success("Ready!"); st.cache_data.clear(); st.rerun()
                api_key, api_target = None, "Ollama"
            else:
                api_target = st.selectbox("API Target", ["OpenAI", "Anthropic", "Google Gemini"])
                
                # Dynamic Key Links
                key_links = {
                    "OpenAI": "https://platform.openai.com/api-keys",
                    "Anthropic": "https://platform.claude.com/settings/keys",
                    "Google Gemini": "https://aistudio.google.com/app/api-keys"
                }
                
                if api_target == "OpenAI": m_list = ["gpt-5.4-nano", "o4-mini", "-- Other --"]
                elif api_target == "Anthropic": m_list = ["claude-haiku-4-5", "claude-sonnet-4-6", "-- Other --"]
                else: m_list = ["gemini-2.0-flash", "gemini-1.5-flash", "-- Other --"]
                
                selected_model = st.selectbox("Model", m_list)
                if selected_model == "-- Other --": selected_model = st.text_input("Custom String")
                
                st.caption(f"ℹ️ [Don't have a key? Get one here]({key_links[api_target]})")
                
                # --- REMEMBER MY KEYS FEATURE ---
                if 'saved_keys' not in st.session_state:
                    st.session_state['saved_keys'] = {}
                
                # Get existing key for this provider if it exists
                default_key = st.session_state['saved_keys'].get(api_target, "")
                
                api_key = st.text_input("Secret API Key", value=default_key, type="password")
                
                # Save the key immediately to session state when it changes
                if api_key:
                    st.session_state['saved_keys'][api_target] = api_key

            st.session_state['engine_config'] = {
                "provider": provider if provider == "Local (Ollama)" else api_target,
                "model": selected_model, "compute": "NVIDIA GPU", "api_key": api_key
            }

            st.divider()
            if st.button("🔄 Start New Interview", use_container_width=True):
                # Preserve Configuration and Saved Keys
                keep = ['engine_config', 'sys_logged', 'selected_voice', 'saved_keys']
                for k in list(st.session_state.keys()):
                    if k not in keep: del st.session_state[k]
                st.rerun()

            with st.expander("🗑️ Danger Zone"):
                if st.button("Delete All Data", type="primary", use_container_width=True):
                    FileManager.cleanup_all_data()
                    HistoryManager.clear_history()
                    # Fully wipe session state keys
                    if 'saved_keys' in st.session_state: del st.session_state['saved_keys']
                    st.success("All data and saved keys cleared!")
                    time.sleep(1)
                    st.rerun()

        st.title("🎙️ AI Interview Coach")

        if st.session_state.get('play_now_bytes'):
            b64 = base64.b64encode(st.session_state['play_now_bytes']).decode()
            n = st.session_state.get('audio_nonce', 0)
            st.components.v1.html(f"""
                <audio id="a_{n}" autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>
                <script>document.getElementById("a_{n}").play();</script>
            """, height=0)
            st.session_state['play_now_bytes'] = None

        tab_coach, tab_history = st.tabs(["🎯 Live Coach", "📈 Session History"])
        
        with tab_coach:
            if 'setup_step' not in st.session_state: st.session_state['setup_step'] = 1
            if 'rounds' not in st.session_state: st.session_state['rounds'] = []
            
            with st.expander("🛠️ Interview Setup Wizard", expanded=(st.session_state['setup_step'] < 3)):
                st.markdown("### 1. Define Target Role")
                c1, c2, c3 = st.columns(3)
                ind, role, sen = c1.text_input("Industry"), c2.text_input("Job Title"), c3.selectbox("Seniority", ["Entry-Level", "Mid-Level", "Senior / Lead", "Executive"])
                
                st.divider()
                st.markdown("### 2. Contextual Data (Optional)")
                c_res, c_jd = st.columns(2)
                res_f, jd_f = c_res.file_uploader("Resume"), c_jd.file_uploader("Job Description")
                st.session_state['resume_text'] = parse_file(res_f) if res_f else ""
                st.session_state['job_desc_text'] = parse_file(jd_f) if jd_f else ""

                if st.button("Generate Interview Rounds", disabled=not (ind and role)):
                    with st.spinner("Structuring..."):
                        p = f"List 4 interview rounds for {sen} {role} in {ind}. Output ONLY a Python list."
                        resp = APIClient.generate_response("Strict list output.", p, [], st.session_state['engine_config'])
                        try:
                            m = re.search(r"\[.*\]", resp, re.DOTALL)
                            st.session_state['rounds'] = ast.literal_eval(m.group()) if m else [resp]
                        except: st.session_state['rounds'] = [resp]
                        st.session_state['setup_step'] = 2; st.rerun()

                if st.session_state['setup_step'] >= 2:
                    st.divider()
                    sel_r = st.selectbox("Stage:", st.session_state['rounds'])
                    sel_p = st.selectbox("Interviewer Style:", list(Personas.PERSONA_PROMPTS.keys()))
                    if st.button("Generate Questions", type="primary"):
                        with st.spinner("Writing..."):
                            st.session_state.update({'selected_persona_label': sel_p, 'selected_round': sel_r})
                            persona_cfg = Personas.get_interviewer_by_type(sel_r.split(" ")[0], sen)
                            st.session_state['round_info'] = {"meaning": persona_cfg['meaning'], "persona": persona_cfg['persona']}
                            q_resp = APIClient.generate_questions(sen, role, ind, sel_r, st.session_state['engine_config'], st.session_state['resume_text'], st.session_state['job_desc_text'])
                            try:
                                m = re.search(r"\[.*\]", q_resp, re.DOTALL)
                                qs = ast.literal_eval(m.group()) if m else [q_resp]
                                st.session_state['custom_questions'] = [clean_llm_text(q) for q in qs]
                            except: st.session_state['custom_questions'] = [clean_llm_text(q_resp)]
                            st.session_state['setup_step'] = 3; st.rerun()

            if st.session_state['setup_step'] == 3:
                st.subheader("🎙️ Live Interview Simulator")
                info = st.session_state['round_info']
                st.info(f"⏱️ **Stage:** {info['meaning']} | **Interviewer:** {info['persona']}")

                if 'chat_history' not in st.session_state:
                    p_lab, r_nam = st.session_state.get('selected_persona_label', 'Standard HR'), st.session_state.get('selected_round', 'General Interview')
                    st.session_state['sys_p'] = Personas.get_interview_sys_prompt(p_lab, r_nam, sen, role, ind, st.session_state['resume_text'], st.session_state['job_desc_text'])
                    st.session_state['chat_history'] = []
                    first_q = st.session_state['custom_questions'][0] if st.session_state['custom_questions'] else "Let's begin."
                    trigger_voice(first_q)
                    st.session_state['chat_history'].append({"role": "assistant", "content": first_q})
                    st.rerun()

                for idx, msg in enumerate(st.session_state['chat_history']):
                    with st.chat_message(msg["role"], avatar=("🤖" if msg["role"] == "assistant" else "👤")):
                        st.write(msg["content"])
                        if msg["role"] == "assistant":
                            if st.button("🔊 Replay Audio", key=f"replay_{idx}"):
                                trigger_voice(msg["content"]); st.rerun()

                st.divider()
                audio_path = record_audio()
                c_sub, c_end = st.columns(2)
                if audio_path and c_sub.button("🗣️ Submit Answer", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        t, m, d, e = APIClient.process_audio(audio_path, "Balanced", "NVIDIA GPU")
                        if not e:
                            st.session_state['chat_history'].append({"role": "user", "content": t})
                            with st.spinner("Thinking..."):
                                nxt = APIClient.generate_response(st.session_state['sys_p'], t, st.session_state['chat_history'][:-1], st.session_state['engine_config'])
                                trigger_voice(nxt)
                                st.session_state['chat_history'].append({"role": "assistant", "content": clean_llm_text(nxt)})
                                st.rerun()
                if c_end.button("🛑 End Interview", use_container_width=True):
                    st.session_state['interview_complete'] = True; st.rerun()

            if st.session_state.get('interview_complete'):
                st.divider(); st.header("📊 Final Analysis"); st.success("Interview Complete!")

        with tab_history:
            history = HistoryManager.load_history()
            if history: st.dataframe(pd.DataFrame(history))
            else: st.info("No history.")

    except Exception as e:
        st.error(f"🚨 Error: {e}"); st.code(traceback.format_exc())

if __name__ == "__main__":
    FileManager.initialize_directories(); main()
