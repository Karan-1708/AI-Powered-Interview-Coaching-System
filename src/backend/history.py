import json
import os
from datetime import datetime

class HistoryManager:
    HISTORY_FILE = os.path.join("temp_data", "session_history.json")

    @staticmethod
    def save_session(wpm, filler_count, tone_label, difficulty_mode):
        """Saves the interview session metrics to a local JSON file."""
        # Ensure the temp_data directory exists
        os.makedirs(os.path.dirname(HistoryManager.HISTORY_FILE), exist_ok=True)
        
        # Load existing history
        history = []
        if os.path.exists(HistoryManager.HISTORY_FILE):
            with open(HistoryManager.HISTORY_FILE, "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
            
        # Append new session record
        record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "wpm": round(wpm, 2),
            "filler_count": filler_count,
            "tone": tone_label,
            "mode": difficulty_mode
        }
        history.append(record)
        
        # Save back to file
        with open(HistoryManager.HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
            
    @staticmethod
    def get_history():
        """Retrieves the session history."""
        if os.path.exists(HistoryManager.HISTORY_FILE):
            with open(HistoryManager.HISTORY_FILE, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []