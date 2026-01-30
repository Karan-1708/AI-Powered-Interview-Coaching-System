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
        Loads the Whisper model with error handling.
        """
        try:
            # 'medium.en' is a great balance of speed/accuracy for the RTX 3090.
            model = WhisperModel("medium.en", device="cuda", compute_type="float16")
            return model
        except Exception as e:
            # We raise the error so the UI can catch it and display a nice message
            raise RuntimeError(f"Failed to load AI Model: {str(e)}")

    def transcribe(self, audio_path):
        """
        Transcribes the audio file.
        Returns: (transcript, processing_time, error_message)
        """
        # 1. Validation Check
        if not os.path.exists(audio_path):
            return None, 0, "Error: Audio file not found on disk."

        start_time = time.time()
        
        try:
            # 2. Load Model (Safe Load)
            model = self.load_model()
            
            # 3. Transcribe
            # Beam size 5 provides better accuracy
            segments, info = model.transcribe(audio_path, beam_size=5)
            
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
                
            processing_time = time.time() - start_time
            
            # Success: Return text, time, and NO error
            return full_text.strip(), processing_time, None

        except RuntimeError as e:
            # Often related to CUDA/GPU running out of memory
            return None, 0, f"GPU Error: {str(e)}"
        
        except Exception as e:
            # Generic catch-all
            return None, 0, f"Unexpected Transcription Error: {str(e)}"