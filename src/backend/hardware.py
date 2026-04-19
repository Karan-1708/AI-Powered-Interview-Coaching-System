import platform
import cpuinfo
import psutil
import subprocess
import shutil
from src.utils.diagnostics import safe_execute

# --- DEFENSIVE IMPORT ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class HardwareInfo:
    def __init__(self):
        self.os_name = platform.system()
        try:
            self.cpu_info = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
        except Exception:
            self.cpu_info = "Unknown CPU"

        # 1. NVIDIA Check
        self.has_nvidia = TORCH_AVAILABLE and torch.cuda.is_available()
        if not self.has_nvidia and shutil.which("nvidia-smi"):
            try:
                subprocess.check_output("nvidia-smi -L", shell=True)
                self.has_nvidia = True
            except Exception:
                pass

        # 2. Apple Silicon Check
        self.is_apple_silicon = False
        if self.os_name == "Darwin":
            cpu_brand = self.cpu_info.lower()
            if "apple" in cpu_brand or "m1" in cpu_brand or "m2" in cpu_brand or "m3" in cpu_brand or "m4" in cpu_brand:
                self.is_apple_silicon = True
        
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)

    @safe_execute(default_val=("CPU & RAM Core", "Checking hardware..."), log_msg="Hardware Recommendation Error")
    def get_recommendation(self):
        """Returns the recommended config and a help string for the user."""
        if self.has_nvidia:
            return "NVIDIA GPU", "🚀 NVIDIA GPU detected. Use this for fastest transcription and high-accuracy models."
        
        if self.is_apple_silicon:
            return "Apple Silicon", "🍎 Apple M-Series chip detected. High performance detected! Use 'Apple Silicon' mode for optimized Neural Engine processing."
        
        if self.total_ram_gb >= 12:
            return "CPU & RAM Core", "💻 Good RAM detected (12GB+). CPU mode will be stable and use 'Small' accuracy models."
        
        return "CPU & RAM Core", "⚠️ Low resources detected. Using CPU mode with 'Tiny' models to ensure stability."

    def get_compute_type(self, tier):
        """Returns the ctranslate2 compute type based on hardware and tier selection."""
        if tier == "NVIDIA GPU": return "float16"
        if tier == "Apple Silicon": return "float32" # Apple Silicon handles float32 best via CPU/NE
        return "int8" # Standard CPU
