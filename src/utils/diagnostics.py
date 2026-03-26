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
            # 1. Gather all potential site-packages locations
            possible_site_packages = site.getsitepackages()
            try: possible_site_packages.append(site.getusersitepackages())
            except: pass
            
            # 2. Add current sys.path entries
            for p in sys.path:
                if "site-packages" in p and p not in possible_site_packages:
                    possible_site_packages.append(p)

            # 3. AGGRESSIVE: Search sibling conda environments
            # This handles cases where libs are in 'ai-interview-coach' but server runs in 'base'
            prefix = sys.prefix
            # Try to find the 'envs' directory
            envs_dirs = []
            if "envs" in prefix:
                envs_dirs.append(os.path.join(prefix.split("envs")[0], "envs"))
            else:
                envs_dirs.append(os.path.join(prefix, "envs"))
            
            for ed in envs_dirs:
                if os.path.exists(ed):
                    for env in os.listdir(ed):
                        possible_site_packages.append(os.path.join(ed, env, "Lib", "site-packages"))

            # 4. Search for NVIDIA bin folders in all gathered locations
            nvidia_bins = []
            for base_path in set(possible_site_packages):
                if not base_path or not os.path.exists(base_path): continue
                
                # Check direct subfolders and 'nvidia' subfolder
                for sub in [
                    os.path.join("nvidia", "cublas", "bin"),
                    os.path.join("nvidia", "cudnn", "bin"),
                    os.path.join("nvidia", "cuda_runtime", "bin"),
                    os.path.join("nvidia", "cuda_nvrtc", "bin"),
                    "bin" # Some older or manual installs
                ]:
                    full_path = os.path.join(base_path, sub)
                    if os.path.exists(full_path):
                        if any(f.endswith(".dll") for f in os.listdir(full_path)):
                            nvidia_bins.append(full_path)

            # 5. Apply DLL directories and update PATH
            if nvidia_bins:
                for bin_path in set(nvidia_bins):
                    if bin_path not in os.environ["PATH"]:
                        os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                    
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(bin_path)
                        except Exception: pass
        except Exception: 
            pass

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
    # Using sys.stdout.buffer to avoid encoding issues on some Windows terminals
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    except Exception:
        # Fallback if stream handler fails
        pass

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
            logger.info("FFmpeg: Detected [OK]")
        else:
            logger.warning("FFmpeg: NOT DETECTED [FAILED] (Some audio formats may fail)")

        # Check for Pillow (Required for PDF images)
        try:
            from PIL import Image
            logger.info(f"Pillow: Detected [OK] (Version: {getattr(Image, '__version__', 'Unknown')})")
        except ImportError:
            logger.warning("Pillow: NOT DETECTED [FAILED] (PDF image support disabled)")
            
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
                import traceback
                # Get the last few lines of the traceback for the log
                tb = traceback.format_exc()
                logger.error(f"{log_msg} in '{func.__name__}': {str(e)}\n{tb}")
                return default_val
        return wrapper
    return decorator