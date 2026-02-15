import os
import sys
import psutil
import logging
import platform
import shutil

# Configure Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "app_debug.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w" # Overwrite log on each restart
)

def log_system_info():
    """Logs critical system stats on startup."""
    try:
        mem = psutil.virtual_memory()
        logging.info(f"OS: {platform.system()} {platform.release()}")
        logging.info(f"Python: {sys.version}")
        logging.info(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        logging.info(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        
        # Check for FFmpeg (Critical for some audio formats)
        if shutil.which("ffmpeg"):
            logging.info("FFmpeg: Detected ✅")
        else:
            logging.warning("FFmpeg: NOT DETECTED ❌ (Some audio formats may fail)")
            
    except Exception as e:
        logging.error(f"Failed to log system info: {e}")

def get_logger():
    return logging.getLogger()