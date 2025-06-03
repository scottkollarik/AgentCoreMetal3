import psutil
import platform
import logging
from typing import Dict, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources including CPU, memory, and GPU usage."""
    
    def __init__(self):
        """Initialize the resource monitor."""
        self._gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            if platform.system() == "Darwin":  # macOS
                # Check for Metal support
                return os.path.exists("/System/Library/Frameworks/Metal.framework")
            elif platform.system() == "Linux":
                # Check for NVIDIA GPU
                return os.path.exists("/dev/nvidia0")
            elif platform.system() == "Windows":
                # Check for NVIDIA GPU on Windows
                return os.path.exists("C:\\Windows\\System32\\nvcuda.dll")
            return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 0.0
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage.
        
        Returns:
            Tuple of (used_memory_percent, used_memory_gb)
        """
        try:
            memory = psutil.virtual_memory()
            return memory.percent, memory.used / (1024 ** 3)  # Convert to GB
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0, 0.0
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get current GPU usage percentage if available."""
        if not self._gpu_available:
            return None
            
        try:
            if platform.system() == "Darwin":  # macOS
                # Use Metal performance shaders to get GPU usage
                # This is a placeholder - actual implementation would use Metal API
                return 0.0
            elif platform.system() in ["Linux", "Windows"]:
                # Use NVIDIA SMI to get GPU usage
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            return None
        except Exception as e:
            logger.error(f"Error getting GPU usage: {e}")
            return None
    
    def get_process_resources(self, pid: int) -> Dict[str, float]:
        """Get resource usage for a specific process.
        
        Args:
            pid: Process ID to monitor
            
        Returns:
            Dictionary containing CPU, memory, and GPU usage
        """
        try:
            process = psutil.Process(pid)
            
            # Get CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            memory_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
            
            # Get GPU usage if available
            gpu_percent = self.get_gpu_usage()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_gb": memory_gb,
                "gpu_percent": gpu_percent
            }
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} not found")
            return {}
        except Exception as e:
            logger.error(f"Error getting process resources: {e}")
            return {}
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get overall system resource usage.
        
        Returns:
            Dictionary containing CPU, memory, and GPU usage
        """
        try:
            cpu_percent = self.get_cpu_usage()
            memory_percent, memory_gb = self.get_memory_usage()
            gpu_percent = self.get_gpu_usage()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_gb": memory_gb,
                "gpu_percent": gpu_percent
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {} 