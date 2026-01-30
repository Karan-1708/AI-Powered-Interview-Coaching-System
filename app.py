import os
import sys
import site

# --- DLL FIX START ---
# 1. Allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def register_nvidia_dlls():
    """
    Actively hunts for the missing 'cublas64_12.dll' in site-packages
    and adds it to the DLL search path.
    """
    if sys.platform != "win32":
        return

    # Get all possible site-packages folders (system and virtualenv)
    possible_paths = site.getsitepackages()
    
    # Also add the local user site-packages just in case
    try:
        possible_paths.append(site.getusersitepackages())
    except:
        pass

    dll_found = False
    
    for base_path in possible_paths:
        # We look for nvidia/cublas/bin and nvidia/cudnn/bin
        cublas_bin = os.path.join(base_path, "nvidia", "cublas", "bin")
        cudnn_bin = os.path.join(base_path, "nvidia", "cudnn", "bin")
        
        # check if the specific problematic file exists here
        if os.path.exists(os.path.join(cublas_bin, "cublas64_12.dll")):
            print(f"DEBUG: Found NVIDIA DLLs at: {cublas_bin}")
            
            # Add to System PATH (Critical for CTranslate2)
            os.environ["PATH"] += os.pathsep + cublas_bin
            os.environ["PATH"] += os.pathsep + cudnn_bin
            
            # Add to Python DLL Directory
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cublas_bin)
                    os.add_dll_directory(cudnn_bin)
                except Exception as e:
                    print(f"Warning: Failed to add_dll_directory: {e}")
            
            dll_found = True
            break # Stop looking once found
            
    if not dll_found:
        print("WARNING: Could not locate nvidia-cublas-cu12 automatically.")
        print("You may need to run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")

# Run the fix immediately
register_nvidia_dlls()
# --- DLL FIX END ---

import streamlit as st
# Import our custom modules
from src.ui.recorder import record_audio
from src.backend.audio_processor import AudioProcessor

# Page Config
st.set_page_config(
    page_title="AI Interview Coach", 
    page_icon="🎙️",
    layout="centered"
)

def main():
    st.title("🎙️ AI Interview Coach")
    st.caption("Phase 2: Audio Pipeline Prototype")

    # Initialize the processor
    processor = AudioProcessor()

    # --- 1. Audio Recording Section ---
    st.divider()
    st.subheader("1. Record Your Answer")
    
    # Call the recorder UI function
    audio_path = record_audio()

    # --- 2. Processing Section ---
    if audio_path:
        st.success(f"Audio captured! Saved securely to local disk.")
        
        # Display audio player for verification
        st.audio(audio_path)

        st.divider()
        st.subheader("2. AI Transcription (Local GPU)")
        
        if st.button("Analyze Audio", type="primary"):
            with st.spinner("Processing on RTX 3090..."):
                # Call the backend to transcribe
                try:
                    transcript, duration = processor.transcribe(audio_path)
                    
                    # Display Result
                    st.markdown("### 📝 Transcript")
                    st.success(transcript)
                    st.caption(f"Processed in {duration:.2f} seconds")
                    
                except RuntimeError as e:
                    st.error("Runtime Error during transcription.")
                    st.error(f"Details: {e}")
                    st.info("Tip: If this is a DLL error, check your terminal for DEBUG messages.")

if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs("temp_data", exist_ok=True)
    main()