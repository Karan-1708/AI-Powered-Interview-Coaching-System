import librosa
import numpy as np
import re

class AcousticScorer:
    def __init__(self):
        # Regex patterns
        self.filler_pattern = re.compile(r'\b(um+?|uh+?|ah+?|hmm+|like|you know|sort of|kind of|i mean|basically|actually)\b', re.IGNORECASE)
        self.stutter_pattern = re.compile(r'\b(\w+)-\1\b', re.IGNORECASE)
        self.repetition_pattern = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
        self.blunder_pattern = re.compile(r'(\.\.\.|scratch that|sorry i mean)', re.IGNORECASE)

        # Thresholds (Kept same as previous step)
        self.THRESHOLDS = {
            "Practice Mode": { "wpm_min": 100, "wpm_max": 170, "max_pauses": 5, "max_fillers": 5, "max_blunders": 3 },
            "Technical / Complex": { "wpm_min": 100, "wpm_max": 130, "max_pauses": 4, "max_fillers": 2, "max_blunders": 1 },
            "Standard Interview": { "wpm_min": 140, "wpm_max": 160, "max_pauses": 2, "max_fillers": 2, "max_blunders": 0 },
            "Presentation": { "wpm_min": 130, "wpm_max": 150, "max_pauses": 1, "max_fillers": 0, "max_blunders": 0 }
        }

    def _assess_content_density(self, duration, word_count):
        """
        Implements the "Soft Logic" for word count targets based on duration.
        """
        tip = None
        
        # logic: +/- 10 seconds buffer to trigger the specific advice
        if 50 <= duration <= 70: # Around 1 Minute
            if word_count < 125: tip = "For a 1-min answer, aim for 125-150 words to show depth."
            elif word_count > 160: tip = "For a 1-min answer, try to be more concise (Target: ~140 words)."
            
        elif 110 <= duration <= 130: # Around 2 Minutes
            if word_count < 200: tip = "For a 2-min answer, aim for 200-250 words."
            elif word_count > 260: tip = "You exceeded the typical 250-word target for this duration."
            
        elif 290 <= duration <= 310: # Around 5 Minutes
            if word_count < 700: tip = "For a 5-min presentation, ensure you cover enough ground (~700 words)."
            
        return tip

    def analyze_audio(self, audio_path, transcript, difficulty="Standard Interview"):
        metrics = {
            "wpm": 0, "pause_count": 0, "filler_count": 0, "blunder_count": 0,
            "duration": 0, "pitch_avg": 0, "pitch_var": 0, "energy_avg": 0,
            "tone_label": "Neutral", "error": None, "feedback": {}
        }
        
        limits = self.THRESHOLDS.get(difficulty, self.THRESHOLDS["Standard Interview"])

        try:
            # --- Audio Analysis ---
            y, sr = librosa.load(audio_path, sr=None)
            total_duration = librosa.get_duration(y=y, sr=sr)
            metrics["duration"] = round(total_duration, 2)

            non_silent_intervals = librosa.effects.split(y, top_db=25, ref=np.max)
            active_time = sum([end - start for start, end in non_silent_intervals]) / sr
            
            if active_time < 0.5:
                metrics["error"] = "Audio too short."
                return metrics

            # --- Tone Analysis ---
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = [pitches[index, t] for t in range(pitches.shape[1]) if (index := magnitudes[:, t].argmax()) and 50 < pitches[index, t] < 300]
            if pitch_values:
                metrics["pitch_avg"], metrics["pitch_var"] = int(np.mean(pitch_values)), int(np.std(pitch_values))
            
            rms = librosa.feature.rms(y=y)[0]
            metrics["energy_avg"] = round(float(np.mean(rms)), 3)

            if metrics["pitch_var"] < 15: metrics["tone_label"] = "🤖 Monotone"
            elif metrics["pitch_var"] > 50: metrics["tone_label"] = "🎭 Dynamic"
            else: metrics["tone_label"] = "😐 Neutral"
            
            # --- Text & WPM Metrics ---
            word_count = len(transcript.split())
            if active_time > 0:
                metrics["wpm"] = int((word_count / active_time) * 60)

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

            # --- Soft Logic: Content Density Check ---
            density_tip = self._assess_content_density(total_duration, word_count)
            if density_tip:
                metrics["feedback"]["density_tip"] = density_tip

            # --- Feedback Logic ---
            if metrics["wpm"] < limits["wpm_min"]:
                metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = f"Too Slow (<{limits['wpm_min']})", "off"
            elif metrics["wpm"] > limits["wpm_max"]:
                if metrics["wpm"] >= 170: metrics["feedback"]["wpm"] = "Risky Pace (170+)"
                else: metrics["feedback"]["wpm"] = f"Too Fast (>{limits['wpm_max']})"
                metrics["feedback"]["wpm_status"] = "off"
            else:
                metrics["feedback"]["wpm"], metrics["feedback"]["wpm_status"] = "Ideal Pace", "normal"

            if metrics["wpm"] > 160 and metrics["pitch_var"] > 60: metrics["tone_label"] = "😰 Rushed"

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
                
            if metrics["tone_label"] == "🤖 Monotone": metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Add Expression", "off"
            elif metrics["tone_label"] == "😰 Rushed": metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Breathe", "off"
            else: metrics["feedback"]["tone"], metrics["feedback"]["tone_status"] = "Good Tone", "normal"

            return metrics

        except Exception as e:
            metrics["error"] = f"Analysis Failed: {str(e)}"
            return metrics