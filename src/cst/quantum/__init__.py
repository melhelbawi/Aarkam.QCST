# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum-Enhanced CST Package
Contextual Spectrum Tokenization with Quantum Information Fusion
Fully standalone - no classical dependencies
"""

# Version and metadata
__version__ = "0.1.0"
__author__ = "Quantum CST Team"
__description__ = "Quantum-Enhanced Contextual Spectrum Tokenization"

# Lazy imports to avoid circular dependencies
try:
    from quantum_cst_config import CSTConfig, QuantumConfig, ConfigPresets
    from quantum_cst_module import QuantumEnhancedCSTModule
    from quantum_cst_transformer import QuantumCSTransformer
    from quantum_fragment_encoder import QuantumFragmentEncoder
    from quantum_information_fuser import QuantumInformationFuser
    from quantum_transformer_utils import QuantumTransformer, collate_quantum_batch
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import quantum modules: {e}")

__all__ = [
    'CSTConfig',
    'QuantumConfig',
    'ConfigPresets',
    'QuantumEnhancedCSTModule',
    'QuantumCSTransformer',
    'QuantumFragmentEncoder',
    'QuantumInformationFuser',
    'QuantumTransformer',
    'collate_quantum_batch',
]

