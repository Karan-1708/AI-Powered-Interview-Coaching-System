import platform
import cpuinfo
import psutil
import subprocess
import shutil
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

# --- DEFENSIVE IMPORT ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class HardwareInfo:
    def __init__(self):
        try:
            self.os_name = platform.system()
            # Handle potential cpuinfo failures
            try:
                self.cpu_info = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
            except Exception:
                self.cpu_info = "Generic CPU"
            
            # 1. Primary check: Torch
            self.has_nvidia = TORCH_AVAILABLE and torch.cuda.is_available()
            
            # 2. Fallback check: nvidia-smi (if torch fails)
            if not self.has_nvidia:
                if shutil.which("nvidia-smi"):
                    try:
                        subprocess.check_output("nvidia-smi -L", shell=True, timeout=2)
                        self.has_nvidia = True 
                    except: pass

            self.is_apple_silicon = platform.processor() == 'arm' and self.os_name == 'Darwin'
            
            # Get Total RAM for recommendations
            try:
                self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                self.total_ram_gb = 8.0 # Safe default
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            self.os_name = "Unknown"
            self.has_nvidia = False
            self.is_apple_silicon = False
            self.total_ram_gb = 4.0

    @safe_execute(default_val=("Balanced (Mid Spec)", "System specs unknown."), log_msg="Hardware Recommendation Error")
    def get_recommendation(self):
        """Returns the recommended Tier based on specs."""
        if self.has_nvidia:
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
                    vram_mb = float(subprocess.check_output(cmd, shell=True, timeout=2).decode("utf-8").strip())
                    vram = vram_mb / 1024
                
                if vram >= 4:
                    return "Pro (High Spec)", "[OK] NVIDIA GPU detected. Pro (High Spec) ready."
            except Exception: pass

        if self.is_apple_silicon:
            return "Balanced (Mid Spec)", "[OK] Apple Silicon detected. Optimized for Neural Engine."
        
        if self.total_ram_gb >= 12:
            return "Balanced (Mid Spec)", "[OK] Good RAM amount (12GB+). Balanced Mode recommended."

        return "Eco (Low Spec)", "[WARN] Low System Resources. Eco Mode recommended for speed."

    @safe_execute(default_val="cpu", log_msg="Optimal Device Error")
    def get_optimal_device(self):
        if TORCH_AVAILABLE and torch.cuda.is_available(): 
            return "cuda"
        return "cpu"

    @safe_execute(default_val="int8", log_msg="Compute Type Error")
    def get_compute_type(self, device):
        if device == "cuda": return "float16"
        if self.is_apple_silicon: return "float32"
        return "int8"