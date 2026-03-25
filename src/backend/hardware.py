import platform
import cpuinfo
import psutil
import subprocess
import shutil

# --- DEFENSIVE IMPORT ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class HardwareInfo:
    def __init__(self):
        self.os_name = platform.system()
        self.cpu_info = cpuinfo.get_cpu_info()['brand_raw']
        
        # 1. Primary check: Torch
        self.has_nvidia = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # 2. Fallback check: nvidia-smi (if torch fails)
        if not self.has_nvidia:
            if shutil.which("nvidia-smi"):
                try:
                    # Just check if we can run it and get a response
                    subprocess.check_output("nvidia-smi -L", shell=True)
                    self.has_nvidia = True 
                except: pass

        self.is_apple_silicon = platform.processor() == 'arm' and self.os_name == 'Darwin'
        
        # Get Total RAM for recommendations
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)

    def get_recommendation(self):
        """
        Returns the recommended Tier based on specs.
        """
        # 1. High-End: NVIDIA GPU with >4GB VRAM
        if self.has_nvidia:
            try:
                # Try torch first for VRAM
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    # Fallback to smi for VRAM check
                    cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
                    vram_mb = float(subprocess.check_output(cmd, shell=True).decode("utf-8").strip())
                    vram = vram_mb / 1024
                
                if vram >= 4:
                    return "Pro (High Spec)", "🟢 NVIDIA GPU detected. Pro (High Spec) ready."
            except: pass

        # 2. Mid-Range: Apple Silicon OR >12GB RAM
        if self.is_apple_silicon:
            return "Balanced (Mid Spec)", "🟢 Apple Silicon detected. Optimized for Neural Engine. Balanced recommended."
        
        if self.total_ram_gb >= 12:
            return "Balanced (Mid Spec)", "🟡 Good RAM amount (12GB+). Balanced Mode recommended."

        # 3. Low-End: Everything else
        return "Eco (Low Spec)", "🔴 Low System Resources. Eco Mode recommended for speed."

    def get_optimal_device(self):
        # We only return 'cuda' if TORCH actually sees it
        if TORCH_AVAILABLE and torch.cuda.is_available(): 
            return "cuda"
        return "cpu"

    def get_compute_type(self, device):
        if device == "cuda": return "float16"
        if self.is_apple_silicon: return "float32"
        return "int8"