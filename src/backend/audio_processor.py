import streamlit as st  # pyright: ignore[reportMissingImports]
from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]
from src.backend.scorer import AcousticScorer
from src.backend.hardware import HardwareInfo
from src.utils.diagnostics import get_logger
import os
import time

logger = get_logger()

class AudioProcessor:
    def __init__(self):
        self.scorer = AcousticScorer()
        self.hw = HardwareInfo()

    def load_model(self, tier="Balanced"):
        """
        Loads model with AUTOMATIC FALLBACK.
        If 'Pro' fails, it retries with 'Eco'.
        """
        device = self.hw.get_optimal_device()
        compute_type = self.hw.get_compute_type(device)
        
        # Map Tiers to Model Sizes
        tier_map = {
            "Eco (Low Spec)": "tiny.en",
            "Balanced (Mid Spec)": "small.en",
            "Pro (High Spec)": "medium.en"
        }
        
        target_model = tier_map.get(tier, "small.en")
        
        try:
            logger.info(f"Attempting to load {target_model} on {device} ({compute_type})...")
            return WhisperModel(target_model, device=device, compute_type=compute_type)
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Catch Out-Of-Memory (OOM) errors specifically
            if "out of memory" in error_msg or "cudnn" in error_msg:
                logger.warning(f"CRASH DETECTED: {target_model} failed. Falling back to Eco Mode.")
                st.toast(f"⚠️ 'Pro' mode failed (Out of VRAM). Switching to Eco Mode...", icon="🛡️")
                
                # FALLBACK: Force CPU and Tiny Model
                return WhisperModel("tiny.en", device="cpu", compute_type="int8")
            else:
                raise e # Re-raise unknown errors

    def process_interview(self, audio_path, difficulty="Standard Interview", tier="Balanced"):
        if not os.path.exists(audio_path):
            return None, None, 0, "Error: Audio file not found."

        # --- NEW: Instant Dead Air Check ---
        is_silent, silence_error = self.check_for_silence(audio_path)
        if is_silent:
            return None, None, 0, silence_error
            
        start_time = time.time()
        
        try:
            # Load model (with self-healing)
            model = self.load_model(tier)
            
            # Transcribe
            segments, info = model.transcribe(
                audio_path, 
                beam_size=5,
                initial_prompt="Umm, I-I think... well, actually... so your... it will delete."
            )
            
            full_text = " ".join([seg.text for seg in segments]).strip()

            # Analyze
            metrics = self.scorer.analyze_audio(audio_path, full_text, difficulty=difficulty)
            
            if metrics.get("error"):
                logger.error(f"Analysis Error: {metrics['error']}")
                return full_text, None, 0, metrics["error"]

            total_time = time.time() - start_time
            logger.info(f"Success: Processed in {total_time:.2f}s")
            
            return full_text, metrics, total_time, None

        except Exception as e:
            logger.error(f"Critical Pipeline Error: {str(e)}")
            return None, None, 0, f"Processing Failed: {str(e)}"

    def check_for_silence(self, audio_path):
        """
        Fast pre-check to ensure the audio actually contains speech.
        Prevents wasting GPU/CPU resources on empty recordings.
        """
        import librosa
        import numpy as np
        
        try:
            # Load audio quickly
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Trim leading/trailing silence (top_db=30 is standard threshold)
            trimmed_audio, _ = librosa.effects.trim(y, top_db=30)
            
            # Calculate duration of actual non-silent audio
            active_duration = len(trimmed_audio) / sr
            
            # If there is less than 1.5 seconds of actual sound, reject it
            if active_duration < 1.5:
                return True, "Voice recording error: System was not able to hear you clearly. Please check your microphone."
                
            return False, None
            
        except Exception as e:
            return True, f"Error reading audio file: {e}"