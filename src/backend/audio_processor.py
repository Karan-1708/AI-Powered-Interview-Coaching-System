import streamlit as st
from faster_whisper import WhisperModel
import os
import time

class AudioProcessor:
    def __init__(self):
        pass

    @st.cache_resource
    def load_model(_self):
        """
        Loads the Whisper model. Cached by Streamlit to avoid reloading on every interaction.
        """
        # REMOVED: st.toast line to prevent CacheReplayClosureError
        
        # 'medium.en' is a great balance of speed/accuracy for the RTX 3090.
        model = WhisperModel("medium.en", device="cuda", compute_type="float16")
        return model

    def transcribe(self, audio_path):
        """
        Transcribes the audio file at the given path.
        """
        if not os.path.exists(audio_path):
            return "Error: Audio file not found.", 0

        start_time = time.time()
        
        # Load model (retrieved from cache if already loaded)
        model = self.load_model()
        
        # Beam size 5 provides better accuracy than default
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
            
        processing_time = time.time() - start_time
        
        return full_text.strip(), processing_time