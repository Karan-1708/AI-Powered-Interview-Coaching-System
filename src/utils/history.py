import json
import os
from datetime import datetime
import traceback
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()
HISTORY_FILE = os.path.join("temp_data", "session_history.json")

class HistoryManager:
    @staticmethod
    @safe_execute(default_val=None, log_msg="Save History Error")
    def save_session(wpm, fillers, tone, mode):
        """Saves core metrics to a local JSON file for progression tracking."""
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

    @staticmethod
    @safe_execute(default_val=[], log_msg="Load History Error")
    def load_history():
        """Loads the session history, handling missing or corrupted files."""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    @staticmethod
    @safe_execute(default_val=None, log_msg="Clear History Error")
    def clear_history():
        """Hunts down and deletes all saved session history data."""
        from src.utils.file_manager import FileManager
        
        # Check the logs directory for any history JSON files and wipe them
        if os.path.exists(FileManager.LOG_DIR):
            for file in os.listdir(FileManager.LOG_DIR):
                if file.endswith('.json'):
                    try:
                        os.remove(os.path.join(FileManager.LOG_DIR, file))
                    except:
                        pass
        
        # Also check the temp directory just in case it was saved there
        if os.path.exists(FileManager.TEMP_DIR):
            for file in os.listdir(FileManager.TEMP_DIR):
                if file.endswith('.json'):
                    try:
                        os.remove(os.path.join(FileManager.TEMP_DIR, file))
                    except:
                        pass
        
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
