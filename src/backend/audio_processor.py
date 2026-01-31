import streamlit as st
from faster_whisper import WhisperModel
from src.backend.scorer import AcousticScorer
import os
import time

class AudioProcessor:
    def __init__(self):
        self.scorer = AcousticScorer()

    @st.cache_resource
    def load_model(_self):
        try:
            model = WhisperModel("medium.en", device="cuda", compute_type="float16")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load AI Model: {str(e)}")

    def process_interview(self, audio_path, difficulty="Intermediate"):
        """
        Runs pipeline with specified difficulty level.
        """
        if not os.path.exists(audio_path):
            return None, None, 0, "Error: Audio file not found."

        start_time = time.time()
        
        try:
            model = self.load_model()
            
            # TRICK: We intentionally use "..." and stutters in the prompt
            # so the model knows it's okay to output them.
            segments, info = model.transcribe(
                audio_path, 
                beam_size=5,
                initial_prompt="Umm, I-I think... well, actually... so your... it will delete."
            )
            
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
            full_text = full_text.strip()

            # Analyze with Difficulty
            metrics = self.scorer.analyze_audio(audio_path, full_text, difficulty=difficulty)
            
            if metrics.get("error"):
                return full_text, None, 0, metrics["error"]

            total_time = time.time() - start_time
            return full_text, metrics, total_time, None

        except RuntimeError as e:
            return None, None, 0, f"GPU Error: {str(e)}"
        except Exception as e:
            return None, None, 0, f"Pipeline Error: {str(e)}"