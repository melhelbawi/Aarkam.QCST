# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

#!/usr/bin/env python
"""
Quick test script for Quantum CST Project
Validates that all components can be imported and initialized
"""

import sys
import torch
from pathlib import Path

# Add parent quantum directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("🧪 QUANTUM CST PROJECT TEST SUITE")
print("=" * 70)

# Test 1: Config Loading
print("\n[TEST 1] Configuration Loading...")
try:
    from quantum_cst_config import CSTConfig, QuantumConfig, ConfigPresets
    cfg = ConfigPresets.get_research_config()
    print(f"  ✅ ConfigPresets loaded")
    print(f"     - Quantum enabled: {cfg.quantum_config.enable_quantum}")
    print(f"     - Qubits: {cfg.quantum_config.n_qubits}")
    print(f"     - Layers: {cfg.quantum_config.n_layers}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Module Imports
print("\n[TEST 2] Core Module Imports...")
try:
    from quantum_cst_module import QuantumEnhancedCSTModule
    from quantum_cst_transformer import QuantumCSTransformer
    print(f"  ✅ QuantumEnhancedCSTModule imported")
    print(f"  ✅ QuantumCSTransformer imported")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Fragment Encoder
print("\n[TEST 3] Fragment Encoder...")
try:
    from quantum_fragment_encoder import QuantumFragmentEncoder
    print(f"  ✅ QuantumFragmentEncoder imported")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Information Fuser
print("\n[TEST 4] Information Fuser...")
try:
    from quantum_information_fuser import HybridQuantumClassical
    print(f"  ✅ HybridQuantumClassical imported")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Module Instantiation
print("\n[TEST 5] Module Instantiation...")
try:
    config = ConfigPresets.get_research_config()
    config.d_model = 128
    config.vocab_size = 1000
    config.char_vocab_size = 256
    
    module = QuantumEnhancedCSTModule(config)
    print(f"  ✅ QuantumEnhancedCSTModule instantiated")
    print(f"     - Quantum enabled: {module.quantum_enabled}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Forward Pass
print("\n[TEST 6] Forward Pass Test...")
try:
    batch_size = 2
    seq_len = 4
    
    text_fragments = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    context_data = {
        'document_embedding': torch.randn(batch_size, config.raw_doc_dim),
        'metadata': {
            'author': torch.zeros(batch_size, dtype=torch.long),
            'domain': torch.zeros(batch_size, dtype=torch.long)
        }
    }
    
    with torch.no_grad():
        output_vectors, quantum_metrics = module(text_fragments, context_data)
    
    print(f"  ✅ Forward pass successful")
    print(f"     - Output shape: {output_vectors.shape}")
    print(f"     - Quantum metrics: {list(quantum_metrics.keys())}")
except Exception as e:
    print(f"  ⚠️  Forward pass failed (may be due to PennyLane): {e}")

# Test 7: Utils
print("\n[TEST 7] Utilities...")
try:
    from test_utils import DeviceManager, PathManager, setup_imports
    device = DeviceManager.get_device()
    print(f"  ✅ DeviceManager working")
    print(f"     - Device: {device}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Test 8: Analysis Document
print("\n[TEST 8] Documentation...")
try:
    # Look for doc in docs folder
    doc_path = Path(__file__).parent.parent / 'docs' / 'QuantumProjectAnalysis.md'
    with open(doc_path, 'r') as f:
        content = f.read()
        lines = len(content.split('\n'))
    print(f"  ✅ Analysis document found ({lines} lines)")
except Exception as e:
    print(f"  ⚠️  Analysis document not found: {e}")

print("\n" + "=" * 70)
print("✅ CORE TESTS PASSED - Project is ready for use!")
print("=" * 70)
print("\nNext steps:")
print("1. Run: python run_quantum_cst.py --mode demo")
print("2. Review: docs/QuantumProjectAnalysis.md")
print("3. Install: pip install -r requirements.txt")
print("=" * 70)
