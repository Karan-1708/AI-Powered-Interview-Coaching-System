import os
import logging
from src.utils.diagnostics import get_logger

logger = get_logger()

class FileManager:
    TEMP_DIR = "temp_data"
    LOG_DIR = "logs"

    @classmethod
    def initialize_directories(cls):
        """Ensures required working directories exist on startup."""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

    @classmethod
    def cleanup_all_data(cls):
        """Deletes all temporary audio files and logs (Privacy Feature) - Windows Safe."""
        deleted_files = 0
        
        # 1. Clean Audio Files
        if os.path.exists(cls.TEMP_DIR):
            for f in os.listdir(cls.TEMP_DIR):
                try:
                    # Skip the history JSON, only delete audio files
                    if f.endswith('.wav') or f.endswith('.mp3'):
                        os.remove(os.path.join(cls.TEMP_DIR, f))
                        deleted_files += 1
                except Exception as e:
                    logger.warning(f"Could not delete temp file {f}: {e}")

        # 2. Clean Log Files (Critical Fix for Windows File Locks)
        if os.path.exists(cls.LOG_DIR):
            logger_instance = logging.getLogger("AI_Coach")
            handlers = logger_instance.handlers[:]
            for handler in handlers:
                handler.close()
                logger_instance.removeHandler(handler)
            
            for f in os.listdir(cls.LOG_DIR):
                try:
                    os.remove(os.path.join(cls.LOG_DIR, f))
                    deleted_files += 1
                except Exception as e:
                    logger.error(f"Could not delete log {f}: {e}")
                    
        return deleted_files

    @classmethod
    def safe_delete_file(cls, file_path):
        """Safely removes a single file without crashing if it doesn't exist."""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")