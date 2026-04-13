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

@st.fragment(run_every=2)
def render_resource_usage():
    """Isolated fragment for high-frequency resource monitoring."""
    hw_status = APIClient.get_hardware_status()
    st.subheader("🖥️ Resource Usage")
    if hw_status:
        stats = hw_status.get("stats", {})
        c_v, r_p = stats.get('cpu_percent', 0), stats.get('ram_percent', 0)
        r_u, r_t = stats.get('ram_used_gb', 0), stats.get('ram_total_gb', 0)
        st.progress(min(max(float(c_v) / 100.0, 0.0), 1.0), text=f"CPU: {c_v}%")
        st.progress(min(max(float(r_p) / 100.0, 0.0), 1.0), text=f"RAM: {r_u}/{r_t} GB")
        if stats.get("gpu_detected"):
            v_p, v_u, v_t = stats.get('vram_percent', 0), stats.get('vram_used_gb', 0), stats.get('vram_total_gb', 0)
            st.progress(min(max(float(v_p) / 100.0, 0.0), 1.0), text=f"GPU VRAM: {v_u}/{v_t} GB")
    else:
        st.error("🔴 Backend: Offline")

@st.fragment(run_every=5)
def render_api_panel(w_n):
    """Dedicated section for AI Engine selection and live connection status."""
    st.subheader("🔌 API & Inference")
    
    providers = ["Ollama (Local)", "OpenAI", "Google Gemini", "Anthropic"]
    default_provider = st.session_state.get('default_provider', "Google Gemini")
    default_idx = providers.index(default_provider)
    provider = st.selectbox("Inference Provider", providers, index=default_idx, key=f"prov_sel_{w_n}")
    
    if provider == "Ollama (Local)":
        models = get_cached_models()
        selected_model = st.selectbox("Local Model", models, key=f"model_sel_{w_n}")
        with st.expander("⬇️ Download Model"):
            tag = st.selectbox("Tag", ["llama3.1:8b", "gemma2:9b", "mistral:7b", "-- Other --"], key=f"pull_tag_{w_n}")
            if tag == "-- Other --": tag = st.text_input("Custom Tag", key=f"pull_custom_{w_n}")
            if st.button("Pull Model", use_container_width=True):
                p = st.progress(0); res = APIClient.pull_model_stream(tag)
                for line in res.iter_lines():
                    if line:
                        d = json.loads(line.decode('utf-8'))
                        if "completed" in d and "total" in d and d["total"] > 0: p.progress(d["completed"]/d["total"])
                st.success("Ready!"); st.cache_data.clear(); st.rerun()
        api_key = None
    else:
        m_map = {"OpenAI": ["gpt-5.4-nano", "gpt-5-nano", "o4-mini", "gpt-4o-mini"], "Anthropic": ["claude-sonnet-4-6","claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-0"], "Google Gemini": ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite"]}
        selected_model = st.selectbox("Model", m_map.get(provider, ["-- Other --"]) + ["-- Other --"], key=f"ext_model_{w_n}")
        if selected_model == "-- Other --": selected_model = st.text_input("Custom Model", key=f"ext_custom_{w_n}")
        
        key_links = {"OpenAI": "https://platform.openai.com/api-keys", "Anthropic": "https://platform.claude.com/settings/keys", "Google Gemini": "https://aistudio.google.com/app/api-keys"}
        st.caption(f"ℹ️ [Get {provider} Key]({key_links.get(provider, '#')})")

        # UNIQUE KEY PER PROVIDER: This ensures fields are isolated and don't mix up keys
        api_key_key = f"api_key_{provider}_{w_n}"
        api_key = st.text_input("Secret API Key", value=st.session_state['saved_keys'].get(provider, ""), type="password", key=api_key_key)

        if api_key != st.session_state['saved_keys'].get(provider):
            st.session_state['saved_keys'][provider] = api_key
            FileManager.save_keys(st.session_state['saved_keys'])
    # Sync Config
    compute_target = st.session_state.get(f"comp_target_{w_n}", "CPU & RAM Core")
    st.session_state['engine_config'] = {"provider": provider, "model": selected_model, "compute": compute_target, "api_key": api_key}

    # Connection Status Badges
    st.write("")
    hw_status = APIClient.get_hardware_status()
    if hw_status:
        st.success("🟢 Backend: Online")
        success, msg = APIClient.test_connection(st.session_state['engine_config'])
        if success: st.success("🟢 AI Engine: Ready")
        else: st.error(msg)
    else:
        st.error("🔴 Backend: Offline")

@st.fragment
def render_config_panel(w_n):
    """Isolated fragment for hardware and voice configuration."""
    st.subheader("⚙️ Configuration")
    
    # 1. Hardware Detection & Selection
    hw = HardwareInfo()
    rec_mode, rec_text = hw.get_recommendation()
    
    # Detect and show physical hardware names
    hw_status = APIClient.get_hardware_status()
    det_hw = hw_status.get("detected_hw", "Standard CPU") if hw_status else "Scanning..."
    
    c_options = ["NVIDIA GPU", "CPU & RAM Core"]
    if hw.is_apple_silicon:
        c_options = ["Apple Silicon", "CPU & RAM Core"]
    
    compute_target = st.radio("Compute Allocation", c_options, horizontal=True, key=f"comp_target_{w_n}")
    
    # Show ALL detected hardware
    st.caption(f"**Detected Hardware:** {det_hw}")
    if hw.has_nvidia and hw.is_apple_silicon: # Rare edge case
        st.caption("🔍 Multiple accelerators found (NVIDIA + M-Series)")
    
    with st.expander("💡 Hardware Helper"):
        st.info(rec_text)

    # 2. Voice Selection
    v_opt = st.radio("Coach Voice", ["Male", "Female"], horizontal=True, key=f"v_opt_{w_n}")
    st.session_state['selected_voice'] = "en-US-GuyNeural" if v_opt == "Male" else "en-US-AvaNeural"

    # 3. Active Mode Logic (Shows real-time backend usage)
    if hw_status:
        actual_using = "Standard CPU (Int8)"
        if compute_target == "NVIDIA GPU" and hw_status.get("has_nvidia"):
            actual_using = "NVIDIA GPU (FP16)"
        elif compute_target == "Apple Silicon" and hw_status.get("is_apple_silicon"):
            actual_using = "Apple Neural Engine (FP32)"
        st.info(f"**Active Mode:** {actual_using}")

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
def render_interview_loop(info):
    """Isolated interview loop for real-time interaction without page flickering."""
    st.subheader("🎙️ Live Interview Simulator")
    st.info(f"⏱️ **Stage:** {info['meaning']} | **Interviewer:** {info['persona']}")
    
    for idx, msg in enumerate(st.session_state['chat_history']):
        with st.chat_message(msg["role"], avatar=("🤖" if msg["role"] == "assistant" else "👤")):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                if st.button("🔊 Replay", key=f"replay_{idx}"):
                    trigger_voice(msg["content"]); st.rerun()

    st.divider()
    if 'rec_nonce' not in st.session_state: st.session_state['rec_nonce'] = 0
    audio_path = record_audio(key=f"recorder_{st.session_state['rec_nonce']}")
    
    c_sub, c_end = st.columns(2)
    compute_type = st.session_state.get('engine_config', {}).get('compute', 'CPU & RAM Core')
    
    if audio_path and c_sub.button("🗣️ Submit Answer", type="primary", use_container_width=True):
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

    if c_end.button("🛑 End Interview & Analyze", use_container_width=True):
        if audio_path:
            with st.spinner("Finalizing..."):
                t, m, d, e = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
                if not e:
                    st.session_state['aggregated_metrics'].append({"transcript": t, "metrics": m, "duration": d})
                    st.session_state['chat_history'].append({"role": "user", "content": t})
        st.session_state['interview_complete'] = True; st.rerun()

@st.fragment
def render_setup_wizard(w_n):
    """Isolated setup wizard for role definition and document processing."""
    with st.expander("🛠️ Interview Setup Wizard", expanded=(st.session_state['setup_step'] < 3)):
        st.markdown("### 1. Define Target Role")
        c1, c2, c3 = st.columns(3)
        ind = c1.text_input("Industry", placeholder="e.g., Tech", key=f"ind_in_{w_n}")
        role = c2.text_input("Job Title", placeholder="e.g., Backend Developer", key=f"role_in_{w_n}")
        sen = c3.selectbox("Seniority", ["Entry-Level", "Mid-Level", "Senior / Lead", "Executive"], key=f"sen_in_{w_n}")
        
        st.divider()
        st.markdown("### 2. Contextual Data (Optional)")
        c_res, c_jd = st.columns(2)
        res_f = c_res.file_uploader("Upload Resume", type=["pdf", "txt", "docx"], key=f"res_up_{w_n}")
        jd_f = c_jd.file_uploader("Upload Job Description", type=["pdf", "txt", "docx"], key=f"jd_up_{w_n}")
        
        if res_f:
            res_text = parse_file(res_f)
            if res_text: st.session_state['resume_text'] = res_text
            else: st.warning(f"⚠️ Could not read resume: {res_f.name}"); st.session_state['resume_text'] = None
        else: st.session_state['resume_text'] = ""

        if jd_f:
            jd_text = parse_file(jd_f)
            if jd_text: st.session_state['job_desc_text'] = jd_text
            else: st.warning(f"⚠️ Could not read job description: {jd_f.name}"); st.session_state['job_desc_text'] = None
        else: st.session_state['job_desc_text'] = ""

        if st.button("Generate Interview Rounds", disabled=not (ind and role), use_container_width=True):
            hw_status = APIClient.get_hardware_status()
            if not hw_status: st.error("🔴 Backend Offline")
            else:
                success, msg = APIClient.test_connection(st.session_state['engine_config'])
                if not success: st.error(f"❌ Connection Error: {msg}")
                else:
                    with st.spinner("Structuring..."):
                        p = (f"List 4 unique interview rounds for a {sen} {role} in the {ind} industry. Output ONLY a Python list of strings.")
                        resp = APIClient.generate_response("Expert recruiter. Return ONLY a Python list.", p, [], st.session_state['engine_config'])
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
            if st.button("Start Interview Simulation", type="primary", use_container_width=True):
                if not sel_r: st.error("Please generate interview rounds in Step 1 first!")
                else:
                    with st.spinner("Writing questions..."):
                        st.session_state.update({'selected_persona_label': sel_p, 'selected_round': sel_r, 'seniority': sen, 'job_title': role, 'industry': ind})
                        persona_cfg = Personas.get_interviewer_by_type(sel_r.split(" ")[0], sen)
                        st.session_state['round_info'] = {"meaning": persona_cfg['meaning'], "persona": persona_cfg['persona'], "recommended_mode": "Balanced"}
                        q_resp = APIClient.generate_questions(sen, role, ind, sel_r, st.session_state['engine_config'], st.session_state['resume_text'], st.session_state['job_desc_text'])
                        raw_qs = [q.strip() for q in q_resp.split('\n') if len(q.strip()) > 10]
                        st.session_state['custom_questions'] = [clean_llm_text(q) for q in raw_qs]
                        st.session_state['setup_step'] = 3; st.rerun()

# --- MAIN APP ---
def main():
    try:
        if 'wipe_nonce' not in st.session_state: st.session_state['wipe_nonce'] = 0
        w_n = st.session_state['wipe_nonce']

        if 'default_provider' not in st.session_state:
            try:
                res = requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
                if res.status_code == 200: st.session_state['default_provider'] = "Ollama (Local)"
                else: st.session_state['default_provider'] = "Google Gemini"
            except: st.session_state['default_provider'] = "Google Gemini"

        if 'saved_keys' not in st.session_state:
            st.session_state['saved_keys'] = FileManager.load_saved_keys()

        # --- SIDEBAR ---
        with st.sidebar:
            st.image("assets/Data-Drifters.png", width="stretch")
            
            # 1. Resource Usage (Top)
            render_resource_usage()
            st.divider()
            
            # 2. API & Inference (Middle)
            render_api_panel(w_n)
            st.divider()
            
            # 3. Configuration (Bottom - Now a Fragment)
            render_config_panel(w_n)
            
            st.divider()
            if st.button("🔄 Start New Interview", use_container_width=True):
                new_nonce = st.session_state.get('wipe_nonce', 0) + 1
                keep = ['engine_config', 'sys_logged', 'selected_voice', 'saved_keys', 'default_provider']
                for k in list(st.session_state.keys()):
                    if k not in keep: del st.session_state[k]
                st.session_state['wipe_nonce'] = new_nonce; st.rerun()

            with st.expander("🗑️ Danger Zone"):
                if st.button("Delete All Data", type="primary", use_container_width=True):
                    FileManager.cleanup_all_data(); HistoryManager.clear_history()
                    FileManager.safe_delete_file(os.path.join(FileManager.TEMP_DIR, "vault.json"))
                    new_nonce = st.session_state.get('wipe_nonce', 0) + 1
                    for k in list(st.session_state.keys()): del st.session_state[k]
                    st.session_state['wipe_nonce'] = new_nonce; st.rerun()

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
            
            if st.session_state['setup_step'] < 3: render_setup_wizard(w_n)

            if st.session_state['setup_step'] == 3 and not st.session_state.get('interview_complete'):
                # (Logic handled in fragments)
                pass

            if st.session_state['setup_step'] == 3 and not st.session_state.get('interview_complete'):
                if 'chat_history' not in st.session_state:
                    st.session_state['sys_p'] = Personas.get_interview_sys_prompt(st.session_state['selected_persona_label'], st.session_state['selected_round'], st.session_state['seniority'], st.session_state['job_title'], st.session_state['industry'], st.session_state['resume_text'], st.session_state['job_desc_text'])
                    st.session_state['chat_history'] = []
                    with st.spinner("🎙️ Coach is entering the room..."):
                        first_q = APIClient.generate_response(st.session_state['sys_p'], "Start the interview. Greet me and ask ONLY your first question.", [], st.session_state['engine_config'], resume_context=st.session_state.get('resume_text', ""), job_context=st.session_state.get('job_desc_text', ""))
                        first_q = clean_llm_text(first_q); trigger_voice(first_q)
                        st.session_state['chat_history'].append({"role": "assistant", "content": first_q}); st.rerun()
                render_interview_loop(st.session_state['round_info'])

            if st.session_state.get('interview_complete'):
                if 'final_feedback' not in st.session_state:
                    with st.spinner("Analyzing..."):
                        v = len(st.session_state['aggregated_metrics'])
                        t_w = sum([m['metrics']['wpm'] for m in st.session_state['aggregated_metrics']])
                        t_f = sum([m['metrics']['filler_count'] for m in st.session_state['aggregated_metrics']])
                        t_d = sum([m['duration'] for m in st.session_state['aggregated_metrics']])
                        
                        # New Metrics Aggregation
                        t_p = sum([m['metrics'].get('pause_count', 0) for m in st.session_state['aggregated_metrics']])
                        t_b = sum([m['metrics'].get('blunder_count', 0) for m in st.session_state['aggregated_metrics']])
                        tones = [m['metrics'].get('tone_label', 'Neutral') for m in st.session_state['aggregated_metrics']]
                        dom_tone = max(set(tones), key=tones.count) if tones else "Neutral"
                        
                        avg_w = t_w / v if v > 0 else 0; full_t = ""
                        for msg in st.session_state['chat_history']: full_t += f"<b>{'Interviewer' if msg['role'] == 'assistant' else 'Candidate'}:</b> {msg['content']}<br><br>\n"
                        f_p = Personas.get_final_feedback_prompt(st.session_state['seniority'], st.session_state['job_title'], st.session_state['industry'], full_t)
                        f_f = APIClient.generate_response(Personas.AI_COACH['system_prompt'], f_p, [], st.session_state['engine_config'], resume_context=st.session_state['resume_text'])
                        
                        # Save session with dominant tone
                        HistoryManager.save_session(avg_w, t_f, dom_tone, "Multi-Turn")
                        
                        st.session_state.update({
                            'final_feedback': f_f, 
                            'full_transcript': full_t, 
                            'avg_wpm': avg_w, 
                            'total_fillers': t_f, 
                            'total_duration': t_d,
                            'total_pauses': t_p,
                            'total_blunders': t_b,
                            'dominant_tone': dom_tone
                        })
                render_final_analysis(st.session_state)

        with tab_history: render_history_dashboard()

    except Exception as e:
        st.error(f"🚨 Error: {e}"); st.code(traceback.format_exc())

if __name__ == "__main__":
    main()