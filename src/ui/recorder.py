import streamlit as st
import time
import os

def record_audio():
    """
    Renders the audio recorder widget and saves the file locally.
    Returns: Path to the saved file (or None if no recording).
    """
    try:
        # Custom CSS for styling
        st.markdown("""
            <style>
            .stAudioInput { width: 100%; }
            </style>
            """, unsafe_allow_html=True)

        st.info("🎙️ Practice Mode: Click the mic to start recording.")
        audio_value = st.audio_input("Record your answer")

        if audio_value:
            # Create a timestamped filename
            timestamp = int(time.time())
            
            # Safe Directory Creation
            try:
                os.makedirs("temp_data", exist_ok=True)
            except OSError as e:
                st.error(f"Failed to create directory: {e}")
                return None

            save_path = f"temp_data/recording_{timestamp}.wav"
            
            # Safe File Writing
            try:
                with open(save_path, "wb") as f:
                    f.write(audio_value.read())
                
                # Verify file size (prevent processing empty files)
                if os.path.getsize(save_path) == 0:
                    st.error("Error: Recorded file is empty.")
                    return None
                    
                return save_path
                
            except IOError as e:
                st.error(f"Failed to save audio file: {e}")
                return None
        
        return None

    except Exception as e:
        st.error(f"critical Recorder Error: {e}")
        return None