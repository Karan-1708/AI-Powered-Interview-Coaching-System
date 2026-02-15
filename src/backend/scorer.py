import librosa  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import re

class AcousticScorer:
    def __init__(self):
        # Regex patterns
        self.filler_pattern = re.compile(r'\b(um+?|uh+?|ah+?|hmm+|like|you know|sort of|kind of|i mean|basically|actually)\b', re.IGNORECASE)
        self.stutter_pattern = re.compile(r'\b(\w+)-\1\b', re.IGNORECASE)
        self.repetition_pattern = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
        self.blunder_pattern = re.compile(r'(\.\.\.|scratch that|sorry i mean)', re.IGNORECASE)

        self.THRESHOLDS = {
            "Practice Mode": { "wpm_min": 100, "wpm_max": 200, "max_pauses": 5, "max_fillers": 5, "max_blunders": 3 },
            "Technical / Complex": { "wpm_min": 100, "wpm_max": 120, "max_pauses": 4, "max_fillers": 2, "max_blunders": 1 },
            "Standard Interview": { "wpm_min": 130, "wpm_max": 160, "max_pauses": 2, "max_fillers": 2, "max_blunders": 0 },
            "Presentation": { "wpm_min": 130, "wpm_max": 150, "max_pauses": 1, "max_fillers": 0, "max_blunders": 0 }
        }

    def _assess_content_density(self, duration, word_count):
        """Soft Logic tip generator."""
        tip = None
        if 50 <= duration <= 70:
            if word_count < 125: tip = "For a 1-min answer, aim for 125-150 words."
            elif word_count > 160: tip = "For a 1-min answer, try to be more concise (Target: ~140 words)."
        elif 110 <= duration <= 130:
            if word_count < 200: tip = "For a 2-min answer, aim for 200-250 words."
            elif word_count > 260: tip = "You exceeded the typical 250-word target."
        return tip

    def analyze_audio(self, audio_path, transcript, difficulty="Standard Interview"):
        metrics = {
            "wpm": 0, "pause_count": 0, "filler_count": 0, "blunder_count": 0,
            "duration": 0, "pitch_avg": 0, "pitch_var": 0, "energy_avg": 0,
            "tone_label": "Neutral", "error": None, "feedback": {}
        }
        
        limits = self.THRESHOLDS.get(difficulty, self.THRESHOLDS["Standard Interview"])

        try:
            # --- 1. Signal Extraction ---
            y, sr = librosa.load(audio_path, sr=None)
            total_duration = librosa.get_duration(y=y, sr=sr)
            metrics["duration"] = round(total_duration, 2)

            non_silent_intervals = librosa.effects.split(y, top_db=25, ref=np.max)
            active_time = sum([end - start for start, end in non_silent_intervals]) / sr
            
            if active_time < 0.5:
                metrics["error"] = "Audio too short."
                return metrics

            # --- 2. Advanced Signal Metrics ---
            
            # PITCH (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = [pitches[index, t] for t in range(pitches.shape[1]) if (index := magnitudes[:, t].argmax()) and 50 < pitches[index, t] < 300]
            if pitch_values:
                metrics["pitch_avg"] = int(np.mean(pitch_values))
                metrics["pitch_var"] = int(np.std(pitch_values)) # Shakiness/Variation
            
            # VOLUME (Energy) - Normalized roughly 0.0 to 0.1+
            rms = librosa.feature.rms(y=y)[0]
            metrics["energy_avg"] = round(float(np.mean(rms)), 3)

            # SPEED (WPM)
            word_count = len(transcript.split())
            minutes = total_duration / 60.0
            metrics["wpm"] = int(word_count / minutes) if minutes > 0 else 0

            # --- 3. EMOTIONAL CLASSIFICATION LOGIC ---
            # Based on your "Common Vocal Indicators Summary" Matrix
            
            # Define Boolean Flags for cleaner logic
            is_fast = metrics["wpm"] > 160
            is_slow = metrics["wpm"] < 110
            is_loud = metrics["energy_avg"] > 0.06  # Thresholds calibrated for typical mic
            is_quiet = metrics["energy_avg"] < 0.02
            is_high_pitch = metrics["pitch_avg"] > 160 # Approx threshold for "strained"
            is_shaky = metrics["pitch_var"] > 40
            is_monotone = metrics["pitch_var"] < 15

            # Priority 1: High Activation (Intense Emotions)
            if is_fast and is_shaky:
                if is_loud:
                    metrics["tone_label"] = "😠 Angry/Intense"
                    metrics["feedback"]["tone"] = "Volume & Pitch high. Too aggressive?"
                    metrics["feedback"]["tone_status"] = "off"
                else:
                    metrics["tone_label"] = "😰 Nervous"
                    metrics["feedback"]["tone"] = "Fast & Shaky. Deep breaths needed."
                    metrics["feedback"]["tone_status"] = "off"
            
            # Priority 2: Positive Activation
            elif is_fast and is_loud and not is_shaky:
                 metrics["tone_label"] = "🤩 Energetic"
                 metrics["feedback"]["tone"] = "Great energy! Passionate delivery."
                 metrics["feedback"]["tone_status"] = "normal"

            # Priority 3: Low Activation / Formal
            elif is_monotone:
                if is_loud:
                    metrics["tone_label"] = "👔 Formal/Stiff"
                    metrics["feedback"]["tone"] = "Authoritative but slightly robotic."
                    metrics["feedback"]["tone_status"] = "normal"
                elif is_quiet:
                    metrics["tone_label"] = "😴 Bored/Sad"
                    metrics["feedback"]["tone"] = "Low energy. Project more voice."
                    metrics["feedback"]["tone_status"] = "off"
                else:
                    metrics["tone_label"] = "🤖 Monotone"
                    metrics["feedback"]["tone"] = "Vary your pitch to engage listeners."
                    metrics["feedback"]["tone_status"] = "off"

            # Priority 4: The Ideal State
            elif not is_fast and not is_slow and not is_quiet:
                 metrics["tone_label"] = "🧘 Calm/Confident"
                 metrics["feedback"]["tone"] = "Steady, resonant, and controlled."
                 metrics["feedback"]["tone_status"] = "normal"

            # Fallback
            else:
                 metrics["tone_label"] = "😐 Casual/Conversational"
                 metrics["feedback"]["tone"] = "Relaxed pace and volume."
                 metrics["feedback"]["tone_status"] = "normal"


            # --- 4. Count Metrics (Fillers/Pauses) ---
            fillers = len(self.filler_pattern.findall(transcript))
            stutters = len(self.stutter_pattern.findall(transcript))
            repetitions = len(self.repetition_pattern.findall(transcript))
            metrics["filler_count"] = fillers + stutters + repetitions
            metrics["blunder_count"] = len(self.blunder_pattern.findall(transcript))

            pause_count = 0
            for i in range(len(non_silent_intervals) - 1):
                gap = (non_silent_intervals[i+1][0] - non_silent_intervals[i][1]) / sr
                if gap > 1.5: pause_count += 1
            metrics["pause_count"] = pause_count

            # --- 5. Final Feedback Compilation ---
            # Soft Logic Tip
            density_tip = self._assess_content_density(total_duration, word_count)
            if density_tip: metrics["feedback"]["density_tip"] = density_tip

            # WPM Feedback
            if metrics["wpm"] < limits["wpm_min"]:
                metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = f"Too Slow (<{limits['wpm_min']})", "off"
            elif metrics["wpm"] > limits["wpm_max"]:
                metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = f"Too Fast (>{limits['wpm_max']})", "off"
            else:
                metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = "Ideal Pace", "normal"

            # Check for Nervous Rushed State specifically
            if metrics["wpm"] > 160 and metrics["pitch_var"] > 50:
                 metrics["tone_label"] = "😰 Nervous!" # Override

            # Count Feedback
            if metrics["pause_count"] <= limits["max_pauses"]:
                metrics["feedback"]["pause"], metrics["feedback"]["pause_status"] = "Good Flow", "normal"
            else:
                metrics["feedback"]["pause"], metrics["feedback"]["pause_status"] = "Too Many Pauses", "off"

            if metrics["filler_count"] <= limits["max_fillers"]:
                metrics["feedback"]["filler"], metrics["feedback"]["filler_status"] = "Clean", "normal"
            else:
                metrics["feedback"]["filler"], metrics["feedback"]["filler_status"] = "Avoid Fillers", "off"

            if metrics["blunder_count"] <= limits["max_blunders"]:
                metrics["feedback"]["blunder"], metrics["feedback"]["blunder_status"] = "Clear Logic", "normal"
            else:
                metrics["feedback"]["blunder"], metrics["feedback"]["blunder_status"] = "Broken Sentences", "off"

            return metrics

        except Exception as e:
            metrics["error"] = f"Analysis Failed: {str(e)}"
            return metrics