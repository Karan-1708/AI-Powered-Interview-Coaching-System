from faster_whisper import WhisperModel
from src.backend.scorer import AcousticScorer
from src.backend.hardware import HardwareInfo
from src.utils.diagnostics import get_logger, safe_execute
import os
import time
import gc

# --- DEFENSIVE IMPORT ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger()

class AudioProcessor:
    _model_cache = {}

    def __init__(self):
        self.scorer = AcousticScorer()
        self.hw = HardwareInfo()

    @safe_execute(default_val=None, log_msg="Whisper Load Error")
    def get_model(self, compute_mode="CPU & RAM Core"):
        """
        Retrieves a cached model or loads a new one based on the selected compute mode.
        """
        # Determine device and model size based on user selection
        if compute_mode == "NVIDIA GPU" and self.hw.has_nvidia:
            device = "cuda"
            target_model_size = "medium.en"  # High accuracy for GPU
            compute_type = "float16"
        else:
            device = "cpu"
            # Adjust CPU model based on RAM
            target_model_size = "small.en" if self.hw.total_ram_gb >= 12 else "tiny.en"
            compute_type = "int8"

        cache_key = f"{target_model_size}_{device}_{compute_type}"

        if cache_key in AudioProcessor._model_cache:
            return AudioProcessor._model_cache[cache_key]

        # Clear cache before loading new model to prevent OOM
        if AudioProcessor._model_cache:
            logger.info(f"Switching compute modes. Clearing cache for {target_model_size}...")
            AudioProcessor._model_cache.clear()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"[INIT] Loading Whisper {target_model_size} on {device} ({compute_type})...")
        model = WhisperModel(target_model_size, device=device, compute_type=compute_type)
        AudioProcessor._model_cache[cache_key] = model
        return model

    @safe_execute(default_val=(None, None, 0, "Processing Error"), log_msg="Interview Processing Error")
    def process_interview(self, audio_path, difficulty="Standard Interview", compute_mode="CPU & RAM Core"):
        if not os.path.exists(audio_path):
            return None, None, 0, "Error: Audio file not found."

        is_silent, silence_error = self.check_for_silence(audio_path)
        if is_silent:
            return None, None, 0, silence_error
            
        start_time = time.time()
        
        # Load model based on user's sidebar selection
        model = self.get_model(compute_mode)
        if not model:
            return None, None, 0, "Error: Could not load Whisper model."
            
        # Transcribe
        logger.info(f"[AUDIO] Transcribing {os.path.basename(audio_path)}...")
        try:
            segments, info = model.transcribe(
                audio_path, 
                beam_size=5,
                initial_prompt="Umm, I-I think... well, actually... so your..."
            )
            # Important: Actually trigger the transcription by iterating (lazy loading)
            full_text = " ".join([seg.text for seg in segments]).strip()
        except RuntimeError as e:
            if "cublas64_12.dll" in str(e) or "cudnn" in str(e):
                logger.critical("CUDA DLL Error: Missing cublas64_12.dll or cudnn.")
                return None, None, 0, "GPU Acceleration Error: Missing NVIDIA DLLs. Please ensure you are running in the 'ai-interview-coach' environment or switch to 'CPU & RAM Core' in the sidebar."
            raise e
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None, None, 0, f"Transcription failed: {str(e)}"

        # Analyze
        metrics = self.scorer.analyze_audio(audio_path, full_text, difficulty=difficulty)
        
        if metrics.get("error"):
            logger.error(f"Analysis Error: {metrics['error']}")
            return full_text, None, 0, metrics["error"]

        total_time = time.time() - start_time
        logger.info(f"[OK] Success: Processed in {total_time:.2f}s")
        
        return full_text, metrics, total_time, None

    @safe_execute(default_val=(True, "Audio check failed"), log_msg="Silence Check Error")
    def check_for_silence(self, audio_path):
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=16000)
        trimmed_audio, _ = librosa.effects.trim(y, top_db=30)
        active_duration = len(trimmed_audio) / sr
        
        if active_duration < 1.0:
            return True, "Microphone did not pick up enough audio. Please try again."
            
        return False, None
