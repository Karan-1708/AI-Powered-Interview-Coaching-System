import os
import logging
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

class FileManager:
    TEMP_DIR = "temp_data"
    LOG_DIR = "logs"
    ASSETS_DIR = "assets"

    @classmethod
    @safe_execute(log_msg="Dir Init Error")
    def initialize_directories(cls):
        """Ensures required working directories exist on startup."""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.ASSETS_DIR, exist_ok=True)

    @classmethod
    @safe_execute(default_val=0, log_msg="Cleanup Error")
    def cleanup_all_data(cls):
        """Deletes all temporary audio files and logs (Privacy Feature) - Windows Safe."""
        deleted_files = 0
        
        # 1. Clean Audio Files
        if os.path.exists(cls.TEMP_DIR):
            for f in os.listdir(cls.TEMP_DIR):
                # Skip the vault.json to preserve keys if we wanted, 
                # but user said 'delete all data' should wipe them.
                try:
                    full_path = os.path.join(cls.TEMP_DIR, f)
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                        deleted_files += 1
                except Exception as e:
                    logger.warning(f"Could not delete temp file {f}: {e}")

        # 2. Clean Log Files
        if os.path.exists(cls.LOG_DIR):
            for f in os.listdir(cls.LOG_DIR):
                if f == "app_debug.log": continue
                try:
                    full_path = os.path.join(cls.LOG_DIR, f)
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                        deleted_files += 1
                except Exception as e:
                    logger.error(f"Could not delete log {f}: {e}")
                    
        return deleted_files

    @classmethod
    @safe_execute(log_msg="Safe Delete Error")
    def safe_delete_file(cls, file_path):
        """Safely removes a single file without crashing if it doesn't exist."""
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    @classmethod
    @safe_execute(default_val={}, log_msg="Key Load Error")
    def load_saved_keys(cls):
        """Loads API keys from a local persistent file."""
        key_file = os.path.join(cls.TEMP_DIR, "vault.json")
        if os.path.exists(key_file):
            import json
            try:
                with open(key_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    @classmethod
    @safe_execute(log_msg="Key Save Error")
    def save_keys(cls, keys_dict):
        """Saves API keys to a local persistent file."""
        key_file = os.path.join(cls.TEMP_DIR, "vault.json")
        import json
        with open(key_file, "w") as f:
            json.dump(keys_dict, f)
