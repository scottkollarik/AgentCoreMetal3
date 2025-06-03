import os
import platform
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class PlatformConfig:
    """Platform configuration manager for AgentCore.
    Handles automatic platform detection and configuration for:
    - MacOS (Metal 3 / CPU)
    - Windows (CUDA / CPU)
    - Linux (CUDA / CPU)
    """
    
    def __init__(self, config_path: str = "configs/deployment_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.platform_info = self._detect_platform()
        self._configure_environment()

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration from YAML file."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _detect_platform(self) -> Dict[str, Any]:
        """Detect current platform and hardware capabilities."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Detect Apple Silicon
        is_apple_silicon = system == 'darwin' and 'arm' in machine
        
        # Detect NVIDIA GPU (simplified - in production, use proper GPU detection)
        has_nvidia = False
        if system in ['linux', 'windows']:
            try:
                import torch
                has_nvidia = torch.cuda.is_available()
            except ImportError:
                pass

        return {
            'system': system,
            'machine': machine,
            'is_apple_silicon': is_apple_silicon,
            'has_nvidia': has_nvidia
        }

    def _configure_environment(self) -> None:
        """Configure environment variables based on platform and configuration."""
        if self.config['platform']['auto_detect']:
            self._configure_for_detected_platform()
        else:
            self._configure_for_forced_platform()

    def _configure_for_detected_platform(self) -> None:
        """Configure environment for automatically detected platform."""
        if self.platform_info['is_apple_silicon']:
            self._configure_macos_metal()
        elif self.platform_info['has_nvidia']:
            self._configure_cuda()
        else:
            self._configure_cpu()

    def _configure_for_forced_platform(self) -> None:
        """Configure environment for forced platform setting."""
        forced_platform = self.config['platform']['force_platform']
        if forced_platform == 'macos':
            self._configure_macos_metal()
        elif forced_platform in ['windows', 'linux']:
            if self.config['hardware'][forced_platform]['cuda']['enabled']:
                self._configure_cuda()
            else:
                self._configure_cpu()

    def _configure_macos_metal(self) -> None:
        """Configure environment for MacOS Metal acceleration."""
        metal_config = self.config['hardware']['macos']['metal']
        if metal_config['enabled']:
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(
                metal_config['memory_management']['high_watermark_ratio']
            )
            if metal_config['memory_management']['shader_cache']:
                os.environ['PYTORCH_MPS_USE_METAL_SHADER_CACHE'] = '1'
            if metal_config['memory_management']['compile_options']:
                os.environ['PYTORCH_MPS_USE_METAL_COMPILE_OPTIONS'] = '1'

    def _configure_cuda(self) -> None:
        """Configure environment for CUDA acceleration."""
        system = self.platform_info['system']
        cuda_config = self.config['hardware'][system]['cuda']
        if cuda_config['enabled']:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{cuda_config['memory_management']['max_split_size_mb']}"

    def _configure_cpu(self) -> None:
        """Configure environment for CPU-only operation."""
        os.environ['PYTORCH_CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable Metal

    def get_platform_info(self) -> Dict[str, Any]:
        """Get current platform information."""
        return self.platform_info

    def get_active_config(self) -> Dict[str, Any]:
        """Get active configuration based on current platform."""
        system = self.platform_info['system']
        if self.platform_info['is_apple_silicon']:
            return self.config['hardware']['macos']
        return self.config['hardware'][system]

# Usage example:
# platform_config = PlatformConfig()
# print(platform_config.get_platform_info())
# print(platform_config.get_active_config()) 