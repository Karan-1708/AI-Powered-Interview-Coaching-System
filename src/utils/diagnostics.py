import os
import sys
import psutil
import logging
import platform
import shutil

# --- 1. GLOBAL CRASH PROTECTION & DLL MAPPING ---
def setup_environment():
    """Centralized environment setup for Windows/NVIDIA support."""
    # Prevent OMP duplication errors
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Programmatically add local 'bin' folder to PATH for portable FFmpeg
    local_bin = os.path.join(os.getcwd(), "bin")
    if os.path.exists(local_bin) and local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = local_bin + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(local_bin)
            except Exception:
                pass

    if platform.system() == "Windows":
        try:
            import site
            # 1. Collect potential base paths
            possible_paths = site.getsitepackages()
            try:
                possible_paths.append(site.getusersitepackages())
            except Exception:
                pass
            
            # Add current environment and typical conda locations
            possible_paths.append(sys.prefix)
            possible_paths.append(os.path.join(sys.prefix, "Library", "bin"))

            # 2. Search for NVIDIA components in both 'bin' and 'lib'
            found_dirs = []
            for base_path in set(possible_paths):
                if not base_path or not os.path.exists(base_path): continue
                
                # Search common nvidia-pip subfolders
                for sub in [
                    os.path.join("nvidia", "cublas", "bin"),
                    os.path.join("nvidia", "cublas", "lib"),
                    os.path.join("nvidia", "cudnn", "bin"),
                    os.path.join("nvidia", "cudnn", "lib"),
                    os.path.join("nvidia", "cuda_runtime", "bin"),
                    "bin"
                ]:
                    sd = os.path.join(base_path, sub)
                    if os.path.exists(sd):
                        # Check if it contains any DLLs
                        if any(f.lower().endswith(".dll") for f in os.listdir(sd)):
                            found_dirs.append(sd)
            
            # 3. Map the found directories
            for sd in set(found_dirs):
                if sd not in os.environ["PATH"]:
                    os.environ["PATH"] = sd + os.pathsep + os.environ["PATH"]
                
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(sd)
                    except Exception as e:
                        logging.getLogger("AI_Coach").debug(f"add_dll_directory failed for {sd}: {e}")
        except Exception as e:
            logging.getLogger("AI_Coach").debug(f"DLL path setup error: {e}", exc_info=True)

# Run this once on import to ensure the environment is ready
setup_environment()

# Configure Logging Directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a custom named logger
logger = logging.getLogger("AI_Coach")
logger.setLevel(logging.INFO)

# Prevent adding multiple handlers if the module is reloaded
if not logger.handlers:
    # 1. File Handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "app_debug.log"), mode="a", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    except Exception as e:
        logger.warning(f"Console logging handler setup failed: {e}")

def log_system_info():
    """Logs critical system stats on startup."""
    try:
        mem = psutil.virtual_memory()
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Python Executable: {sys.executable}")
        logger.info(f"Architecture: {platform.architecture()[0]}")
        logger.info(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        
        # Check for FFmpeg (System or Local bin folder)
        local_ffmpeg = os.path.join(os.getcwd(), "bin", "ffmpeg.exe")
        if shutil.which("ffmpeg") or os.path.exists(local_ffmpeg):
            logger.info("FFmpeg: Detected [OK]")
        else:
            logger.warning("FFmpeg: NOT DETECTED [FAILED]")

        # Check for CUDA availability in torch
        try:
            import torch
            torch_ver = getattr(torch, "__version__", "Unknown")
            logger.info(f"PyTorch Version: {torch_ver}")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA: Detected [OK] (Device: {torch.cuda.get_device_name(0)})")
            else:
                logger.warning("CUDA: NOT DETECTED by Torch (Check DLLs, Drivers, or Version)")
        except ImportError:
            logger.warning("PyTorch not installed")
            
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
                tb = traceback.format_exc()
                logger.error(f"{log_msg} in '{func.__name__}': {str(e)}\n{tb}")
                return default_val
        return wrapper
    return decorator
