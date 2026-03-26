import psutil
import platform
import logging
import subprocess
import shutil
import re

from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ResourceMonitor:
    def __init__(self):
        # Initial call to seed cpu_percent
        try:
            psutil.cpu_percent(interval=None)
        except Exception: pass

    @safe_execute(default_val=(None, 0, 0, 0), log_msg="GPU SMI Error")
    def _get_gpu_stats_via_smi(self):
        """
        Fallback: Use nvidia-smi to get GPU info if torch is not working.
        Returns (name, used_gb, total_gb, percent) or (None, 0, 0, 0)
        """
        if not shutil.which("nvidia-smi"):
            return None, 0, 0, 0

        try:
            # Get Name, Memory Used, and Memory Total
            cmd = "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits"
            result = subprocess.check_output(cmd, shell=True, timeout=2).decode("utf-8").strip()

            if not result:
                return None, 0, 0, 0

            parts = [p.strip() for p in result.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                used_mb = float(parts[1])
                total_mb = float(parts[2])

                used_gb = round(used_mb / 1024, 1)
                total_gb = round(total_mb / 1024, 1)
                percent = int((used_mb / total_mb) * 100) if total_mb > 0 else 0

                return name, used_gb, total_gb, percent
        except Exception:
            pass

        return None, 0, 0, 0

    @safe_execute(default_val={}, log_msg="System Usage Error")
    def get_system_usage(self):
        """
        Returns a dictionary of current current system resources.
        """
        try:
            stats = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "ram_percent": psutil.virtual_memory().percent,
                "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 1),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "gpu_name": None,
                "vram_used_gb": 0,
                "vram_total_gb": 0,
                "vram_percent": 0,
                "gpu_detected": False
            }
        except Exception as e:
            logger.error(f"Failed to get basic system stats: {e}")
            return {}

        # 1. Try Torch first (Active monitoring)
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    # We have a GPU and Torch can see it
                    stats["gpu_detected"] = True
                    stats["gpu_name"] = torch.cuda.get_device_name(0)

                    # mem_get_info returns (free, total) in bytes
                    free_res, total_res = torch.cuda.mem_get_info(0)
                    used_res = total_res - free_res

                    stats["vram_total_gb"] = round(total_res / (1024**3), 1)
                    stats["vram_used_gb"] = round(used_res / (1024**3), 1)

                    if total_res > 0:
                        stats["vram_percent"] = int((used_res / total_res) * 100)
                    else:
                        stats["vram_percent"] = 0
                    return stats # Success with torch
            except Exception:
                pass

        # 2. Fallback to nvidia-smi if torch failed or isn't available
        name, used, total, pct = self._get_gpu_stats_via_smi()
        if name:
            stats["gpu_detected"] = True
            stats["gpu_name"] = name
            stats["vram_used_gb"] = used
            stats["vram_total_gb"] = total
            stats["vram_percent"] = pct

        return stats