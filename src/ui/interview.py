import streamlit as st
import ast
import re
from src.api.client import APIClient
from src.backend.personas import Personas
from src.ui.recorder import record_audio
from src.utils.text_processor import clean_llm_text, parse_file
from src.utils.diagnostics import get_logger

logger = get_logger()


def trigger_voice(text):
    """Fetches TTS audio and stores it in session_state for autoplay."""
    if not text:
        return False
    clean_text = clean_llm_text(text)
    voice = st.session_state.get('selected_voice', "en-US-GuyNeural")
    audio_bytes = APIClient.generate_speech(clean_text, voice=voice)
    if audio_bytes:
        st.session_state['play_now_bytes'] = audio_bytes
        st.session_state['audio_nonce'] = st.session_state.get('audio_nonce', 0) + 1
        return True
    return False


@st.fragment
def render_setup_wizard(w_n):
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
            if res_text:
                st.session_state['resume_text'] = res_text
            else:
                st.warning(f"⚠️ Could not read resume: {res_f.name}")
                st.session_state['resume_text'] = None
        else:
            st.session_state['resume_text'] = ""

        if jd_f:
            jd_text = parse_file(jd_f)
            if jd_text:
                st.session_state['job_desc_text'] = jd_text
            else:
                st.warning(f"⚠️ Could not read job description: {jd_f.name}")
                st.session_state['job_desc_text'] = None
        else:
            st.session_state['job_desc_text'] = ""

        if st.button("Generate Interview Rounds", disabled=not (ind and role), use_container_width=True):
            hw_status = APIClient.get_hardware_status()
            if not hw_status:
                st.error("🔴 Backend Offline")
            else:
                success, msg = APIClient.test_connection(st.session_state['engine_config'])
                if not success:
                    st.error(f"❌ Connection Error: {msg}")
                else:
                    with st.spinner("Structuring..."):
                        p = (f"List 4 unique interview rounds for a {sen} {role} in the {ind} industry. "
                             f"Output ONLY a Python list of strings.")
                        resp = APIClient.generate_response(
                            "Expert recruiter. Return ONLY a Python list.", p, [],
                            st.session_state['engine_config']
                        )
                        try:
                            m = re.search(r"\[.*\]", resp, re.DOTALL)
                            rounds = ast.literal_eval(m.group()) if m else [resp]
                            st.session_state['rounds'] = [str(r).strip() for r in rounds if len(str(r)) < 150]
                        except Exception:
                            st.session_state['rounds'] = ["1. Initial Screen", "2. Technical Round", "3. Culture Fit", "4. Final Manager"]
                        st.session_state['setup_step'] = 2
                        st.rerun()

        if st.session_state['setup_step'] >= 2:
            st.divider()
            st.markdown("### 3. Select Stage & Persona")
            sel_r = st.selectbox("Target Interview Round:", st.session_state['rounds'], key=f"round_sel_{w_n}")
            sel_p = st.selectbox("Interviewer Style:", list(Personas.PERSONA_PROMPTS.keys()), key=f"persona_sel_{w_n}")
            if st.button("Start Interview Simulation", type="primary", use_container_width=True):
                if not sel_r:
                    st.error("Please generate interview rounds in Step 1 first!")
                else:
                    with st.spinner("Writing questions..."):
                        st.session_state.update({
                            'selected_persona_label': sel_p,
                            'selected_round': sel_r,
                            'seniority': sen,
                            'job_title': role,
                            'industry': ind
                        })
                        persona_cfg = Personas.get_interviewer_by_type(sel_r.split(" ")[0], sen)
                        st.session_state['round_info'] = {
                            "meaning": persona_cfg['meaning'],
                            "persona": persona_cfg['persona'],
                            "recommended_mode": "Balanced"
                        }
                        q_resp = APIClient.generate_questions(
                            sen, role, ind, sel_r,
                            st.session_state['engine_config'],
                            st.session_state['resume_text'],
                            st.session_state['job_desc_text']
                        )
                        raw_qs = [q.strip() for q in q_resp.split('\n') if len(q.strip()) > 10]
                        st.session_state['custom_questions'] = [clean_llm_text(q) for q in raw_qs]
                        st.session_state['setup_step'] = 3
                        st.rerun()


@st.fragment
def render_interview_loop(info):
    st.subheader("🎙️ Live Interview Simulator")
    st.info(f"⏱️ **Stage:** {info['meaning']} | **Interviewer:** {info['persona']}")

    for idx, msg in enumerate(st.session_state['chat_history']):
        with st.chat_message(msg["role"], avatar=("🤖" if msg["role"] == "assistant" else "👤")):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                if st.button("🔊 Replay", key=f"replay_{idx}"):
                    trigger_voice(msg["content"])
                    st.rerun()

    st.divider()
    if 'rec_nonce' not in st.session_state:
        st.session_state['rec_nonce'] = 0
    audio_path = record_audio(key=f"recorder_{st.session_state['rec_nonce']}")

    c_sub, c_end = st.columns(2)
    compute_type = st.session_state.get('engine_config', {}).get('compute', 'CPU & RAM Core')

    if audio_path and c_sub.button("🗣️ Submit Answer", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            t, m, d, e = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
        if e:
            st.error(f"Audio Error: {e}")
        else:
            st.session_state['aggregated_metrics'].append({"transcript": t, "metrics": m, "duration": d})
            st.session_state['chat_history'].append({"role": "user", "content": t})
            with st.chat_message("assistant", avatar="🤖"):
                nxt = st.write_stream(APIClient.stream_response(
                    st.session_state['sys_p'], t,
                    st.session_state['chat_history'][:-1],
                    st.session_state['engine_config'],
                    resume_context=st.session_state.get('resume_text', ""),
                    job_context=st.session_state.get('job_desc_text', "")
                ))
            nxt = clean_llm_text(nxt)
            trigger_voice(nxt)
            st.session_state['chat_history'].append({"role": "assistant", "content": nxt})
            st.session_state['rec_nonce'] += 1
            st.rerun()

    if c_end.button("🛑 End Interview & Analyze", use_container_width=True):
        if audio_path:
            with st.spinner("Finalizing..."):
                t, m, d, e = APIClient.process_audio(audio_path, info['recommended_mode'], compute_type)
                if not e:
                    st.session_state['aggregated_metrics'].append({"transcript": t, "metrics": m, "duration": d})
                    st.session_state['chat_history'].append({"role": "user", "content": t})
        st.session_state['interview_complete'] = True
        st.rerun()
