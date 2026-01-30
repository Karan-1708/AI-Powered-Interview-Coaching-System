import os
import sys
import site

# --- CONFIGURATION / DLL FIX START ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def register_nvidia_dlls():
    """
    Actively hunts for the missing 'cublas64_12.dll' in site-packages
    and adds it to the DLL search path.
    """
    if sys.platform != "win32": return

    try:
        possible_paths = site.getsitepackages()
        try: possible_paths.append(site.getusersitepackages())
        except: pass

        for base_path in possible_paths:
            cublas_bin = os.path.join(base_path, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(base_path, "nvidia", "cudnn", "bin")
            
            if os.path.exists(os.path.join(cublas_bin, "cublas64_12.dll")):
                print(f"DEBUG: Found NVIDIA DLLs at: {cublas_bin}")
                os.environ["PATH"] += os.pathsep + cublas_bin
                os.environ["PATH"] += os.pathsep + cudnn_bin
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(cublas_bin)
                    os.add_dll_directory(cudnn_bin)
                break
    except Exception:
        pass # Silent fail here, main app will catch real errors

register_nvidia_dlls()
# --- DLL FIX END ---

import streamlit as st
from src.ui.recorder import record_audio
from src.backend.audio_processor import AudioProcessor

st.set_page_config(page_title="AI Interview Coach", page_icon="🎙️", layout="centered")

def main():
    st.title("🎙️ AI Interview Coach")
    st.caption("Phase 2: Audio Pipeline (Robust)")

    try:
        # Initialize processor (Lazy load)
        processor = AudioProcessor()

        # --- 1. Audio Recording ---
        st.divider()
        st.subheader("1. Record Your Answer")
        
        audio_path = record_audio()

        # --- 2. Processing ---
        if audio_path:
            st.success(f"Audio captured! Saved securely to local disk.")
            st.audio(audio_path)

            st.divider()
            st.subheader("2. AI Transcription (Local GPU)")
            
            if st.button("Analyze Audio", type="primary"):
                with st.spinner("Processing on RTX 3090..."):
                    # Unpack the 3 values: Text, Duration, and Error
                    transcript, duration, error = processor.transcribe(audio_path)
                
                # Handle Results
                if error:
                    st.error(f"Analysis Failed: {error}")
                    st.info("Tip: Check your GPU memory or try restarting the app.")
                else:
                    st.markdown("### Transcript")
                    st.success(transcript)
                    st.caption(f"Processed in {duration:.2f} seconds")

    except Exception as e:
        # Global App Crash Handler
        st.error("Critical Application Error")
        st.code(str(e))
        st.warning("Please restart the application.")

if __name__ == "__main__":
    os.makedirs("temp_data", exist_ok=True)
    main()