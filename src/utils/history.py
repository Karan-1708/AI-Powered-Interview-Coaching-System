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
            "wpm": round(float(wpm), 1),
            "fillers": int(fillers),
            "tone": tone,
            "mode": mode
        }
        
        history = HistoryManager.load_history()
        history.append(entry)
        
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)
            logger.info(f"Session history successfully saved for mode: {mode}")
        except Exception as e:
            logger.error(f"Failed to write history to disk: {e}")
            raise

    @staticmethod
    @safe_execute(default_val=[], log_msg="Load History Error")
    def load_history():
        """Loads the session history, handling missing or corrupted files."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"History file corrupted or unreadable: {e}")
                return []
        return []

    @staticmethod
    @safe_execute(default_val=None, log_msg="Clear History Error")
    def clear_history():
        """Hunts down and deletes all saved session history data."""
        from src.utils.file_manager import FileManager
        
        # Helper to safely remove JSON files from specific directories
        def clean_json_from_dir(directory):
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.json'):
                        try:
                            os.remove(os.path.join(directory, file))
                        except Exception as e:
                            logger.warning(f"Failed to delete {file} from {directory}: {e}")

        clean_json_from_dir(FileManager.LOG_DIR)
        clean_json_from_dir(FileManager.TEMP_DIR)
        
        if os.path.exists(HISTORY_FILE):
            try:
                os.remove(HISTORY_FILE)
            except Exception as e:
                logger.error(f"Failed to delete core history file: {e}")
