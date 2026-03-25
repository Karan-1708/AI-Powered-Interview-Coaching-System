import os
import sys
import psutil
import logging
import platform
import shutil

# --- 1. GLOBAL CRASH PROTECTION & DLL MAPPING ---
def setup_environment():
    """Centralized environment setup for Windows/NVIDIA support."""
    # Prevent OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    if platform.system() == "Windows":
        try:
            import site
            possible_paths = site.getsitepackages()
            try: possible_paths.append(site.getusersitepackages())
            except: pass
            for base_path in possible_paths:
                cublas_bin = os.path.join(base_path, "nvidia", "cublas", "bin")
                cudnn_bin = os.path.join(base_path, "nvidia", "cudnn", "bin")
                if os.path.exists(os.path.join(cublas_bin, "cublas64_12.dll")):
                    os.environ["PATH"] += os.pathsep + cublas_bin
                    os.environ["PATH"] += os.pathsep + cudnn_bin
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(cublas_bin)
                        os.add_dll_directory(cudnn_bin)
                    break
        except Exception: pass

# Run this once on import to ensure the environment is ready
setup_environment()

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

        # Check for Pillow (Required for PDF images)
        try:
            from PIL import Image
            logger.info(f"Pillow: Detected ✅ (Version: {getattr(Image, '__version__', 'Unknown')})")
        except ImportError:
            logger.warning("Pillow: NOT DETECTED ❌ (PDF image support disabled)")
            
    except Exception as e:
        logger.error(f"Failed to log system info: {e}")

def get_logger():
    return logging.getLogger("AI_Coach")

def safe_execute(default_val=None, log_msg="Execution Error"):
    """Decorator for standardized error handling and logging."""
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{log_msg}: {e}", exc_info=True)
                return default_val
        return wrapper
    return decorator