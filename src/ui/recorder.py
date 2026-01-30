import streamlit as st
import time
import os

def record_audio():
    """
    Renders the audio recorder widget and saves the file locally.
    Returns: Path to the saved file (or None if no recording).
    """
    # Custom CSS to make the recorder look professional
    st.markdown("""
        <style>
        .stAudioInput { width: 100%; }
        </style>
        """, unsafe_allow_html=True)

    st.info("🎙️ Practice Mode: Click the mic to start recording.")
    audio_value = st.audio_input("Record your answer")

    if audio_value:
        # Create a timestamped filename to avoid overwriting
        timestamp = int(time.time())
        # Ensure directory exists just in case
        os.makedirs("temp_data", exist_ok=True)
        
        save_path = f"temp_data/recording_{timestamp}.wav"
        
        # Save the bytes to a file
        with open(save_path, "wb") as f:
            f.write(audio_value.read())
            
        return save_path
    
    return None