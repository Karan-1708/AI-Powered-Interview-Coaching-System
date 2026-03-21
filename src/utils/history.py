import json
import os
from datetime import datetime
import traceback
from src.utils.diagnostics import get_logger

logger = get_logger()
HISTORY_FILE = os.path.join("temp_data", "session_history.json")

class HistoryManager:
    @staticmethod
    def save_session(wpm, fillers, tone, mode):
        """Saves core metrics to a local JSON file for progression tracking."""
        try:
            # Safely ensure the directory exists based on the file path
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "wpm": wpm,
                "fillers": fillers,
                "tone": tone,
                "mode": mode
            }
            
            history = HistoryManager.load_history()
            history.append(entry)
            
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)
                
            logger.info(f"Session history successfully saved for mode: {mode}")
            
        except Exception as e:
            logger.error(f"Failed to save history: {e}\n{traceback.format_exc()}")

    @staticmethod
    def load_history():
        """Loads the session history, handling missing or corrupted files."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("History file corrupted. Creating a fresh history array.")
                return []
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                return []
        return []