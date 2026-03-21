import os
import sys
import psutil
import logging
import platform
import shutil

# Configure Logging Directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a custom named logger
logger = logging.getLogger("AI_Coach")
logger.setLevel(logging.INFO)

# Prevent adding multiple handlers if the module is reloaded by Streamlit/FastAPI
if not logger.handlers:
    # 1. File Handler (Saves to logs/app_debug.log)
    file_handler = logging.FileHandler(os.path.join(log_dir, "app_debug.log"), mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (Prints to your Terminal for live debugging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def log_system_info():
    """Logs critical system stats on startup."""
    try:
        mem = psutil.virtual_memory()
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        logger.info(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        
        # Check for FFmpeg (Critical for Whisper audio parsing)
        if shutil.which("ffmpeg"):
            logger.info("FFmpeg: Detected ✅")
        else:
            logger.warning("FFmpeg: NOT DETECTED ❌ (Some audio formats may fail)")
            
    except Exception as e:
        logger.error(f"Failed to log system info: {e}")

def get_logger():
    return logging.getLogger("AI_Coach")