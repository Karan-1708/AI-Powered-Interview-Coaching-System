import streamlit as st
import json
from src.api.client import APIClient
from src.backend.hardware import HardwareInfo
from src.utils.file_manager import FileManager
from src.utils.diagnostics import get_logger

logger = get_logger()


@st.cache_data(ttl=300)
def get_cached_models():
    return APIClient.get_local_models()


@st.fragment(run_every=2)
def render_resource_usage():
    hw_status = APIClient.get_hardware_status()
    st.subheader("🖥️ Resource Usage")
    if hw_status:
        stats = hw_status.get("stats", {})
        c_v, r_p = stats.get('cpu_percent', 0), stats.get('ram_percent', 0)
        r_u, r_t = stats.get('ram_used_gb', 0), stats.get('ram_total_gb', 0)
        st.progress(min(max(float(c_v) / 100.0, 0.0), 1.0), text=f"CPU: {c_v}%")
        st.progress(min(max(float(r_p) / 100.0, 0.0), 1.0), text=f"RAM: {r_u}/{r_t} GB")
        if stats.get("gpu_detected"):
            v_p = stats.get('vram_percent', 0)
            v_u = stats.get('vram_used_gb', 0)
            v_t = stats.get('vram_total_gb', 0)
            st.progress(min(max(float(v_p) / 100.0, 0.0), 1.0), text=f"GPU VRAM: {v_u}/{v_t} GB")
    else:
        st.error("🔴 Backend: Offline")


@st.fragment
def render_api_panel(w_n):
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
            if tag == "-- Other --":
                tag = st.text_input("Custom Tag", key=f"pull_custom_{w_n}")
            if st.button("Pull Model", use_container_width=True):
                p = st.progress(0)
                res = APIClient.pull_model_stream(tag)
                for line in res.iter_lines():
                    if line:
                        d = json.loads(line.decode('utf-8'))
                        if "completed" in d and "total" in d and d["total"] > 0:
                            p.progress(d["completed"] / d["total"])
                st.success("Ready!")
                st.cache_data.clear()
                st.rerun()
        api_key = None
    else:
        m_map = {
            "OpenAI": ["gpt-5.4-nano", "gpt-5-nano", "o4-mini", "gpt-4o-mini"],
            "Anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-0"],
            "Google Gemini": ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite"]
        }
        selected_model = st.selectbox("Model", m_map.get(provider, ["-- Other --"]) + ["-- Other --"], key=f"ext_model_{w_n}")
        if selected_model == "-- Other --":
            selected_model = st.text_input("Custom Model", key=f"ext_custom_{w_n}")

        key_links = {
            "OpenAI": "https://platform.openai.com/api-keys",
            "Anthropic": "https://platform.claude.com/settings/keys",
            "Google Gemini": "https://aistudio.google.com/app/api-keys"
        }
        st.caption(f"ℹ️ [Get {provider} Key]({key_links.get(provider, '#')})")

        api_key = st.text_input(
            "Secret API Key",
            value=st.session_state['saved_keys'].get(provider, ""),
            type="password",
            key=f"api_key_{provider}_{w_n}"
        )
        if api_key != st.session_state['saved_keys'].get(provider):
            st.session_state['saved_keys'][provider] = api_key
            FileManager.save_keys(st.session_state['saved_keys'])

    compute_target = st.session_state.get(f"comp_target_{w_n}", "CPU & RAM Core")
    st.session_state['engine_config'] = {
        "provider": provider,
        "model": selected_model,
        "compute": compute_target,
        "api_key": api_key
    }

    # --- CONNECTION STATUS ---
    # Only call test_connection when provider/model/key actually changes.
    # The fingerprint detects real changes; cached result is shown otherwise.
    fingerprint = f"{provider}|{selected_model}|{api_key or ''}"
    force_retest = st.session_state.pop('force_conn_retest', False)

    if fingerprint != st.session_state.get('conn_fingerprint') or force_retest:
        # Skip testing cloud providers with no key entered yet
        should_test = provider == "Ollama (Local)" or bool(api_key)
        if should_test:
            hw_status = APIClient.get_hardware_status()
            if hw_status:
                success, msg = APIClient.test_connection(st.session_state['engine_config'])
                st.session_state['conn_result'] = ("online", success, msg)
            else:
                st.session_state['conn_result'] = ("offline", False, "")
        else:
            st.session_state['conn_result'] = ("no_key", False, "")
        st.session_state['conn_fingerprint'] = fingerprint

    st.write("")
    result = st.session_state.get('conn_result')
    if result:
        state, success, msg = result
        if state == "offline":
            st.error("🔴 Backend: Offline")
        elif state == "no_key":
            st.info("🔑 Enter your API key to verify the connection.")
        else:
            st.success("🟢 Backend: Online")
            if success:
                st.success("🟢 AI Engine: Ready")
            else:
                st.error(msg)

    if st.button("🔄 Re-test Connection", use_container_width=True):
        st.session_state['force_conn_retest'] = True
        st.rerun()


@st.fragment
def render_config_panel(w_n):
    st.subheader("⚙️ Configuration")

    hw = HardwareInfo()
    _, rec_text = hw.get_recommendation()

    hw_status = APIClient.get_hardware_status()
    det_hw = hw_status.get("detected_hw", "Standard CPU") if hw_status else "Scanning..."

    c_options = ["Apple Silicon", "CPU & RAM Core"] if hw.is_apple_silicon else ["NVIDIA GPU", "CPU & RAM Core"]
    compute_target = st.radio("Compute Allocation", c_options, horizontal=True, key=f"comp_target_{w_n}")

    st.caption(f"**Detected Hardware:** {det_hw}")
    if hw.has_nvidia and hw.is_apple_silicon:
        st.caption("🔍 Multiple accelerators found (NVIDIA + M-Series)")

    with st.expander("💡 Hardware Helper"):
        st.info(rec_text)

    v_opt = st.radio("Coach Voice", ["Male", "Female"], horizontal=True, key=f"v_opt_{w_n}")
    st.session_state['selected_voice'] = "en-US-GuyNeural" if v_opt == "Male" else "en-US-AvaNeural"

    if hw_status:
        actual_using = "Standard CPU (Int8)"
        if compute_target == "NVIDIA GPU" and hw_status.get("has_nvidia"):
            actual_using = "NVIDIA GPU (FP16)"
        elif compute_target == "Apple Silicon" and hw_status.get("is_apple_silicon"):
            actual_using = "Apple Neural Engine (FP32)"
        st.info(f"**Active Mode:** {actual_using}")
