import os
import base64
import traceback
import requests
import streamlit as st

from src.utils.diagnostics import get_logger, log_system_info
from src.api.client import APIClient
from src.utils.file_manager import FileManager
from src.utils.history import HistoryManager
from src.backend.personas import Personas
from src.utils.text_processor import clean_llm_text
from src.ui.sidebar import render_resource_usage, render_api_panel, render_config_panel
from src.ui.interview import render_interview_loop, render_setup_wizard, trigger_voice
from src.ui.dashboard import render_final_analysis, render_history_dashboard

logger = get_logger()


def _run_final_analysis():
    """Aggregates session metrics and generates the final AI feedback report."""
    metrics = st.session_state['aggregated_metrics']
    v = len(metrics)
    t_w = sum(m['metrics']['wpm'] for m in metrics)
    t_f = sum(m['metrics']['filler_count'] for m in metrics)
    t_d = sum(m['duration'] for m in metrics)
    t_p = sum(m['metrics'].get('pause_count', 0) for m in metrics)
    t_b = sum(m['metrics'].get('blunder_count', 0) for m in metrics)
    tones = [m['metrics'].get('tone_label', 'Neutral') for m in metrics]
    dom_tone = max(set(tones), key=tones.count) if tones else "Neutral"
    avg_w = t_w / v if v > 0 else 0

    full_t = ""
    for msg in st.session_state['chat_history']:
        label = 'Interviewer' if msg['role'] == 'assistant' else 'Candidate'
        full_t += f"<b>{label}:</b> {msg['content']}<br><br>\n"

    f_p = Personas.get_final_feedback_prompt(
        st.session_state['seniority'],
        st.session_state['job_title'],
        st.session_state['industry'],
        full_t
    )
    f_f = APIClient.generate_response(
        Personas.AI_COACH['system_prompt'], f_p, [],
        st.session_state['engine_config'],
        resume_context=st.session_state['resume_text']
    )

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


def main():
    try:
        if 'sys_logged' not in st.session_state:
            log_system_info()
            st.session_state['sys_logged'] = True

        FileManager.initialize_directories()

        if 'wipe_nonce' not in st.session_state:
            st.session_state['wipe_nonce'] = 0
        w_n = st.session_state['wipe_nonce']

        if 'default_provider' not in st.session_state:
            try:
                res = requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
                st.session_state['default_provider'] = "Ollama (Local)" if res.status_code == 200 else "Google Gemini"
            except Exception:
                st.session_state['default_provider'] = "Google Gemini"

        if 'saved_keys' not in st.session_state:
            st.session_state['saved_keys'] = FileManager.load_saved_keys()

        # --- SIDEBAR ---
        with st.sidebar:
            st.image("assets/Data-Drifters.png", use_container_width=True)
            render_resource_usage()
            st.divider()
            render_api_panel(w_n)
            st.divider()
            render_config_panel(w_n)
            st.divider()

            if st.button("🔄 Start New Interview", use_container_width=True):
                new_nonce = st.session_state.get('wipe_nonce', 0) + 1
                keep = {'engine_config', 'sys_logged', 'selected_voice', 'saved_keys', 'default_provider'}
                for k in list(st.session_state.keys()):
                    if k not in keep:
                        del st.session_state[k]
                st.session_state['wipe_nonce'] = new_nonce
                st.rerun()

            with st.expander("🗑️ Danger Zone"):
                if st.button("Delete All Data", type="primary", use_container_width=True):
                    FileManager.cleanup_all_data()
                    HistoryManager.clear_history()
                    FileManager.safe_delete_file(os.path.join(FileManager.TEMP_DIR, "vault.json"))
                    new_nonce = st.session_state.get('wipe_nonce', 0) + 1
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.session_state['wipe_nonce'] = new_nonce
                    st.rerun()

        # --- MAIN CONTENT ---
        st.title("🎙️ AI Interview Coach")

        _audio = st.session_state.get('play_now_bytes')
        if _audio and isinstance(_audio, bytes):
            b64 = base64.b64encode(_audio).decode()
            n = st.session_state.get('audio_nonce', 0)
            st.components.v1.html(
                f'<audio id="a_{n}" autoplay>'
                f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                f'<script>document.getElementById("a_{n}").play();</script>',
                height=1
            )
            st.session_state['play_now_bytes'] = None

        tab_coach, tab_history = st.tabs(["🎯 Live Coach", "📈 Session History"])

        with tab_coach:
            if 'setup_step' not in st.session_state:
                st.session_state['setup_step'] = 1
            if 'rounds' not in st.session_state:
                st.session_state['rounds'] = []
            if 'aggregated_metrics' not in st.session_state:
                st.session_state['aggregated_metrics'] = []

            if st.session_state['setup_step'] < 3:
                render_setup_wizard(w_n)

            if st.session_state['setup_step'] == 3 and not st.session_state.get('interview_complete'):
                if 'chat_history' not in st.session_state:
                    st.session_state['sys_p'] = Personas.get_interview_sys_prompt(
                        st.session_state['selected_persona_label'],
                        st.session_state['selected_round'],
                        st.session_state['seniority'],
                        st.session_state['job_title'],
                        st.session_state['industry'],
                        st.session_state['resume_text'],
                        st.session_state['job_desc_text']
                    )
                    st.session_state['chat_history'] = []
                    with st.chat_message("assistant", avatar="🤖"):
                        first_q = st.write_stream(APIClient.stream_response(
                            st.session_state['sys_p'],
                            "Start the interview. Greet me and ask ONLY your first question.",
                            [],
                            st.session_state['engine_config'],
                            resume_context=st.session_state.get('resume_text', ""),
                            job_context=st.session_state.get('job_desc_text', "")
                        ))
                    first_q = clean_llm_text(first_q)
                    trigger_voice(first_q)
                    st.session_state['chat_history'].append({"role": "assistant", "content": first_q})
                    st.rerun()
                render_interview_loop(st.session_state['round_info'])

            if st.session_state.get('interview_complete'):
                if 'final_feedback' not in st.session_state:
                    with st.spinner("Analyzing..."):
                        _run_final_analysis()
                render_final_analysis(st.session_state)

        with tab_history:
            render_history_dashboard()

    except Exception as e:
        st.error(f"🚨 Error: {e}")
        st.code(traceback.format_exc())
