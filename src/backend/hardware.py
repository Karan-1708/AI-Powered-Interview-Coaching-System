import platform
import torch
import cpuinfo
import psutil

class HardwareInfo:
    def __init__(self):
        self.os_name = platform.system()
        self.cpu_info = cpuinfo.get_cpu_info()['brand_raw']
        self.has_nvidia = torch.cuda.is_available()
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
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram >= 4:
                    return "Pro (High Spec)", "🟢 NVIDIA GPU detected. Pro (High Spec) ready."
            except: pass

        # 2. Mid-Range: Apple Silicon OR >12GB RAM
        if self.is_apple_silicon:
            return "Balanced (Mid Spec)", "🟢 Apple Silicon detected. Optimized for Neural Engine. Balanced recommended but do give Pro Mode a try if your computer can handle it."
        
        if self.total_ram_gb >= 12:
            return "Balanced (Mid Spec)", "🟡 Good RAM amount (12GB+). Balanced Mode recommended."

        # 3. Low-End: Everything else
        return "Eco (Low Spec)", "🔴 Low System Resources. Eco Mode recommended for speed."

    def get_optimal_device(self):
        if self.has_nvidia: return "cuda"
        return "cpu"

    def get_compute_type(self, device):
        if device == "cuda": return "float16"
        if self.is_apple_silicon: return "float32"
        return "int8"