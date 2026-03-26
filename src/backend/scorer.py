import librosa
import numpy as np
import re
import logging
from src.utils.diagnostics import safe_execute

logger = logging.getLogger("AI_Coach")

class AcousticScorer:
    # --- CONSTANTS ---
    THRESHOLDS = {
        "Practice Mode": { "wpm_min": 100, "wpm_max": 200, "max_pauses": 5, "max_fillers": 5, "max_blunders": 3 },
        "Technical / Complex": { "wpm_min": 100, "wpm_max": 120, "max_pauses": 4, "max_fillers": 2, "max_blunders": 1 },
        "Standard Interview": { "wpm_min": 130, "wpm_max": 160, "max_pauses": 2, "max_fillers": 2, "max_blunders": 0 },
        "Presentation": { "wpm_min": 130, "wpm_max": 150, "max_pauses": 1, "max_fillers": 0, "max_blunders": 0 }
    }

    FILLER_PATTERN = re.compile(r'\b(um+?|uh+?|ah+?|hmm+|like|you know|sort of|kind of|i mean|basically|actually)\b', re.IGNORECASE)
    STUTTER_PATTERN = re.compile(r'\b(\w+)-\1\b', re.IGNORECASE)
    REPETITION_PATTERN = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
    BLUNDER_PATTERN = re.compile(r'(\.\.\.|scratch that|sorry i mean)', re.IGNORECASE)

    def __init__(self):
        pass

    def _assess_content_density(self, duration, word_count):
        """Soft Logic tip generator."""
        if 50 <= duration <= 70:
            if word_count < 125: return "For a 1-min answer, aim for 125-150 words."
            if word_count > 160: return "For a 1-min answer, try to be more concise (Target: ~140 words)."
        elif 110 <= duration <= 130:
            if word_count < 200: return "For a 2-min answer, aim for 200-250 words."
            if word_count > 260: return "You exceeded the typical 250-word target."
        return None

    @safe_execute(default_val={"error": "Analysis failed"}, log_msg="Audio Analysis Error")
    def analyze_audio(self, audio_path, transcript, difficulty="Standard Interview"):
        metrics = {
            "wpm": 0, "pause_count": 0, "filler_count": 0, "blunder_count": 0,
            "duration": 0, "pitch_avg": 0, "pitch_var": 0, "energy_avg": 0,
            "tone_label": "Neutral", "error": None, "feedback": {}
        }
        
        limits = self.THRESHOLDS.get(difficulty, self.THRESHOLDS["Standard Interview"])

        try:
            # --- 1. Efficient Loading ---
            # Load at 16k for analysis to save memory/CPU
            y, sr = librosa.load(audio_path, sr=16000)
            total_duration = librosa.get_duration(y=y, sr=sr)
            metrics["duration"] = round(total_duration, 2)

            # --- 2. Signal Metrics ---
            # Active time check using simple energy threshold
            non_silent_intervals = librosa.effects.split(y, top_db=25)
            active_time = sum([end - start for start, end in non_silent_intervals]) / sr
            
            if active_time < 0.5:
                metrics["error"] = "Audio too short."
                return metrics

            # PITCH (F0) - Using YIN (Faster and more stable than piptrack)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                metrics["pitch_avg"] = int(np.mean(f0))
                metrics["pitch_var"] = int(np.std(f0))
            
            # VOLUME (RMS Energy)
            rms = librosa.feature.rms(y=y)[0]
            metrics["energy_avg"] = round(float(np.mean(rms)), 3)

            # SPEED (WPM)
            words = transcript.split()
            word_count = len(words)
            minutes = total_duration / 60.0
            metrics["wpm"] = int(word_count / minutes) if minutes > 0 else 0

            # --- 3. TONE CLASSIFICATION ---
            self._classify_tone(metrics)

            # --- 4. Count Metrics (Regex) ---
            fillers = len(self.FILLER_PATTERN.findall(transcript))
            stutters = len(self.STUTTER_PATTERN.findall(transcript))
            repetitions = len(self.REPETITION_PATTERN.findall(transcript))
            metrics["filler_count"] = fillers + stutters + repetitions
            metrics["blunder_count"] = len(self.BLUNDER_PATTERN.findall(transcript))

            # Pauses (> 1.5s)
            pause_count = 0
            for i in range(len(non_silent_intervals) - 1):
                gap = (non_silent_intervals[i+1][0] - non_silent_intervals[i][1]) / sr
                if gap > 1.5: pause_count += 1
            metrics["pause_count"] = pause_count

            # --- 5. Feedback Compilation ---
            self._compile_feedback(metrics, limits, total_duration, word_count)

            return metrics

        except Exception as e:
            logger.error(f"Scorer Error: {e}")
            metrics["error"] = f"Analysis Failed: {str(e)}"
            return metrics

    @safe_execute(default_val=None, log_msg="Tone Classification Error")
    def _classify_tone(self, metrics):
        """Pure logic for emotional labeling."""
        # Logic here...
        # (Assuming the original implementation is fine, just wrapping it)
        is_fast = metrics["wpm"] > 160
        is_slow = metrics["wpm"] < 110
        is_loud = metrics["energy_avg"] > 0.06
        is_quiet = metrics["energy_avg"] < 0.02
        is_shaky = metrics["pitch_var"] > 40
        is_monotone = metrics["pitch_var"] < 15

        if is_fast and is_shaky:
            if is_loud:
                metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Angry/Intense", "Volume & Pitch high. Too aggressive?", "off"
            else:
                metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Nervous", "Fast & Shaky. Deep breaths needed.", "off"
        elif is_fast and is_loud and not is_shaky:
             metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Energetic", "Great energy! Passionate delivery.", "normal"
        elif is_monotone:
            if is_loud:
                metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Formal/Stiff", "Authoritative but slightly robotic.", "normal"
            elif is_quiet:
                metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Bored/Sad", "Low energy. Project more voice.", "off"
            else:
                metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Monotone", "Vary your pitch to engage listeners.", "off"
        elif not is_fast and not is_slow and not is_quiet:
             metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Calm/Confident", "Steady, resonant, and controlled.", "normal"
        else:
             metrics["tone_label"], metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Casual/Conversational", "Relaxed pace and volume.", "normal"

    @safe_execute(default_val=None, log_msg="Feedback Compilation Error")
    def _compile_feedback(self, metrics, limits, duration, word_count):
        """Aggregates all feedback strings."""
        density_tip = self._assess_content_density(duration, word_count)
        if density_tip: metrics["feedback"]["density_tip"] = density_tip

        # WPM
        if metrics["wpm"] < limits["wpm_min"]:
            metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = f"Too Slow (<{limits['wpm_min']})", "off"
        elif metrics["wpm"] > limits["wpm_max"]:
            metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = f"Too Fast (>{limits['wpm_max']})", "off"
        else:
            metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = "Ideal Pace", "normal"

        # Counts
        metrics["feedback"]["pause"], metrics["feedback"]["pause_status"] = ("Good Flow", "normal") if metrics["pause_count"] <= limits["max_pauses"] else ("Too Many Pauses", "off")
        metrics["feedback"]["filler"], metrics["feedback"]["filler_status"] = ("Clean", "normal") if metrics["filler_count"] <= limits["max_fillers"] else ("Avoid Fillers", "off")
        metrics["feedback"]["blunder"], metrics["feedback"]["blunder_status"] = ("Clear Logic", "normal") if metrics["blunder_count"] <= limits["max_blunders"] else ("Broken Sentences", "off")