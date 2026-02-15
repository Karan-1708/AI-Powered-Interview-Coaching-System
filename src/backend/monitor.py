import psutil
import torch
import platform

class ResourceMonitor:
    def get_system_usage(self):
        """
        Returns a dictionary of current system resources.
        """
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 1),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "gpu_name": None,
            "vram_used_gb": 0,
            "vram_total_gb": 0,
            "vram_percent": 0
        }

        # Check for NVIDIA GPU
        if torch.cuda.is_available():
            try:
                stats["gpu_name"] = torch.cuda.get_device_name(0)
                # VRAM calculations (in bytes -> GB)
                total_mem = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                
                stats["vram_total_gb"] = round(total_mem / (1024**3), 1)
                stats["vram_used_gb"] = round(reserved / (1024**3), 1)
                stats["vram_percent"] = int((reserved / total_mem) * 100)
            except:
                pass # Fail silently if driver issues
        
        return stats