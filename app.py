import streamlit as st
import torch
import librosa
import numpy as np
from faster_whisper import WhisperModel

# Page Config
st.set_page_config(page_title="AI Coach - System Check", page_icon="⚙️")

st.title("⚙️ AI Interview Coach: System Diagnostic")

# 1. Check GPU Availability (Crucial for Latency)
st.subheader("1. Hardware Acceleration")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.success(f"GPU Detected: {gpu_name}")
    st.info(f"CUDA Version: {torch.version.cuda}")
else:
    st.error("No GPU Detected. System will run slowly on CPU.")

# 2. Check Libraries
st.subheader("2. Dependency Check")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Librosa:** {librosa.__version__}")
with col2:
    st.write(f"**NumPy:** {np.__version__}")
with col3:
    st.write(f"**Torch:** {torch.__version__}")

# 3. Test Model Loading (Simulated)
st.subheader("3. Model Engine Test")
if st.button("Initialize Whisper Engine"):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        with st.spinner(f"Loading Faster-Whisper on {device}..."):
            # We assume a small model for the test to be quick, or we can use 'tiny' just for a health check
            model = WhisperModel("tiny", device=device, compute_type=compute_type)
        
        st.success(f"Whisper Engine initialized successfully on {device}!")
    except Exception as e:
        st.error(f"Failed to initialize Whisper: {e}")

st.divider()
st.caption("Phase 1: Environment Setup Complete")