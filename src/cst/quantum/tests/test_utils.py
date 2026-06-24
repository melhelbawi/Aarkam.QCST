# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Test Utilities for Quantum CST
Common utilities for testing - device management, imports, and setup
"""

import sys
import os
import logging
import torch
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage device placement (CPU/GPU) for quantum and classical operations"""
    
    _device: Optional[torch.device] = None
    _use_cuda: bool = False
    
    @classmethod
    def get_device(cls, prefer_cuda: bool = True) -> torch.device:
        """
        Get the appropriate device for computation.
        
        Args:
            prefer_cuda: If True, use CUDA if available
            
        Returns:
            torch.device: CPU or CUDA device
        """
        if cls._device is None:
            if prefer_cuda and torch.cuda.is_available():
                cls._device = torch.device("cuda")
                cls._use_cuda = True
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                cls._device = torch.device("cpu")
                cls._use_cuda = False
                logger.info("Using CPU for computation")
        
        return cls._device
    
    @classmethod
    def set_device(cls, device: torch.device):
        """Explicitly set the device to use"""
        cls._device = device
        cls._use_cuda = device.type == "cuda"
    
    @classmethod
    def is_cuda(cls) -> bool:
        """Check if using CUDA"""
        return cls._use_cuda
    
    @classmethod
    def reset(cls):
        """Reset device management"""
        cls._device = None
        cls._use_cuda = False


class PathManager:
    """Manage relative imports between quantum and classical modules"""
    
    _QUANTUM_DIR: Optional[Path] = None
    _CLASSICAL_DIR: Optional[Path] = None
    _setup_done: bool = False
    
    @classmethod
    def setup_paths(cls) -> None:
        """Setup and configure paths for module imports"""
        if cls._setup_done:
            return
        
        # Determine directories
        current_file = Path(__file__).resolve()
        tests_dir = current_file.parent
        cls._QUANTUM_DIR = tests_dir.parent
        cls._CLASSICAL_DIR = cls._QUANTUM_DIR.parent / 'classical'
        
        # Add quantum directory to path if not already there
        quantum_str = str(cls._QUANTUM_DIR)
        if quantum_str not in sys.path:
            sys.path.insert(0, quantum_str)
            logger.debug(f"Added {quantum_str} to sys.path")
        
        cls._setup_done = True
    
    @classmethod
    def get_quantum_dir(cls) -> Path:
        """Get quantum module directory"""
        cls.setup_paths()
        return cls._QUANTUM_DIR
    
    @classmethod
    def get_classical_dir(cls) -> Path:
        """Get classical module directory"""
        cls.setup_paths()
        return cls._CLASSICAL_DIR


def setup_imports():
    """
    Setup imports to allow test modules to import from quantum modules.
    Call this at test initialization.
    """
    PathManager.setup_paths()


def safe_import(module_name: str, package: str = 'quantum'):
    """
    Safely import a module from quantum or classical package.
    
    Args:
        module_name: Name of the module to import (e.g., 'fragment_encoder')
        package: 'quantum' or 'classical'
        
    Returns:
        The imported module or None if not found
    """
    try:
        if package == 'quantum':
            return __import__(module_name)
        elif package == 'classical':
            PathManager.setup_paths()
            return __import__(module_name)
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        return None


class QuantumFallback:
    """Context manager for graceful quantum-to-classical fallback"""
    
    def __init__(self, fallback_fn=None, verbose=True):
        """
        Initialize fallback context.
        
        Args:
            fallback_fn: Function to call if quantum operation fails
            verbose: Whether to log fallback events
        """
        self.fallback_fn = fallback_fn
        self.verbose = verbose
        self.fell_back = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.fell_back = True
            if self.verbose:
                logger.warning(f"Quantum operation failed, falling back to classical: {exc_val}")
            
            if self.fallback_fn is not None:
                try:
                    self.fallback_fn()
                except Exception as e:
                    logger.error(f"Fallback also failed: {e}")
                    raise
            
            return True  # Suppress exception
        
        return False


# Initialize on import
setup_imports()

__all__ = [
    'DeviceManager',
    'PathManager',
    'QuantumFallback',
    'setup_imports',
    'safe_import',
    'logger',
]
