# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

import unittest
import torch
import sys
import os
import shutil

# Add source directories to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src', 'cst', 'quantum'))
sys.path.append(os.path.join(project_root, 'src', 'cst', 'models'))

from quantum_cst_config import CSTConfig, QuantumConfig, ConfigPresets
from quantum_cst_module import QuantumEnhancedCSTModule
from quantum_cst_transformer import QuantumCSTransformer

class TestQuantumCSTConfig(unittest.TestCase):
    def setUp(self):
        self.test_config_dir = "test_configs"
        os.makedirs(self.test_config_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_config_dir):
            shutil.rmtree(self.test_config_dir)

    def test_config_initialization(self):
        config = CSTConfig()
        self.assertIsInstance(config.quantum_config, QuantumConfig)
        self.assertTrue(config.quantum_config.enable_quantum)
        self.assertEqual(config.quantum_config.n_qubits, 8)

    def test_presets(self):
        research = ConfigPresets.get_research_config()
        self.assertEqual(research.quantum_config.n_qubits, 8)
        self.assertTrue(research.quantum_config.log_circuit_depth)

        production = ConfigPresets.get_production_config()
        self.assertEqual(production.quantum_config.n_qubits, 6)
        self.assertFalse(production.quantum_config.track_quantum_metrics)

        high_acc = ConfigPresets.get_high_accuracy_config()
        self.assertEqual(high_acc.quantum_config.n_qubits, 10)
        self.assertEqual(high_acc.quantum_config.entanglement_pattern, 'full')

    def test_save_load_yaml(self):
        config = CSTConfig()
        config.d_model = 128
        config.quantum_config.n_qubits = 4
        
        filepath = os.path.join(self.test_config_dir, "test_config.yaml")
        config.to_yaml(filepath)
        
        loaded_config = CSTConfig.from_yaml(filepath)
        self.assertEqual(loaded_config.d_model, 128)
        self.assertEqual(loaded_config.quantum_config.n_qubits, 4)

    def test_quantum_advantage_estimation(self):
        config = CSTConfig()
        advantage = config.estimate_quantum_advantage()
        self.assertIn('quantum_state_dimension', advantage)
        self.assertIn('theoretical_speedup', advantage)


class TestQuantumCSTModule(unittest.TestCase):
    def setUp(self):
        self.config = CSTConfig()
        self.config.d_model = 32
        self.config.vocab_size = 100
        self.config.char_vocab_size = 50
        self.config.quantum_config = QuantumConfig()
        self.config.quantum_config.enable_quantum = True
        self.config.quantum_config.n_qubits = 4  # Small for testing
        self.config.quantum_config.n_layers = 1
        
        # Mock dependencies to avoid complex setup if needed
        # but here we use real classes as we added paths

    def test_initialization(self):
        module = QuantumEnhancedCSTModule(self.config)
        self.assertIsNotNone(module.information_fuser)
        self.assertTrue(module.quantum_enabled)

    def test_forward_pass(self):
        module = QuantumEnhancedCSTModule(self.config)
        module.enable_profiling_mode(True)
        
        batch_size = 2
        seq_len = 4
        
        text_fragments = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        fragment_chars = torch.randint(0, self.config.char_vocab_size, (batch_size, seq_len, 8))
        context_chars = torch.randint(0, self.config.char_vocab_size, (batch_size, seq_len, 8))
        
        context_data = {
            'document_embedding': torch.randn(batch_size, self.config.raw_doc_dim),
            'metadata': {} # Optional
        }
        
        output, metrics = module(text_fragments, context_data, fragment_chars, context_chars)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))
        self.assertIn('quantum_tokens_in_batch', metrics)
        
        stats = module.get_performance_stats()
        self.assertIn('cache_hits', stats)


class TestQuantumCSTransformer(unittest.TestCase):
    def setUp(self):
        self.config = CSTConfig()
        self.config.d_model = 32
        self.config.num_layers = 2
        self.config.num_heads = 4
        self.config.vocab_size = 100
        self.config.batch_size = 2
        self.config.quantum_config = QuantumConfig()
        self.config.quantum_config.n_qubits = 4
        self.config.quantum_config.n_layers = 1
        self.config.quantum_config.use_quantum_attention = False # Keep simple

    def test_mlm_forward(self):
        model = QuantumCSTransformer(self.config, task_type='mlm')
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        context_data = {'document_embedding': torch.randn(batch_size, self.config.raw_doc_dim)}
        labels = input_ids.clone()

        output = model(input_ids, context_data, labels=labels)
        
        self.assertIn('logits', output)
        self.assertIn('loss', output)
        self.assertEqual(output['logits'].shape, (batch_size, seq_len, self.config.vocab_size))

    def test_classification_forward(self):
        self.config.num_labels = 3
        model = QuantumCSTransformer(self.config, task_type='classification')
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        context_data = {'document_embedding': torch.randn(batch_size, self.config.raw_doc_dim)}
        labels = torch.randint(0, 3, (batch_size,))

        output = model(input_ids, context_data, labels=labels)
        
        self.assertIn('logits', output)
        self.assertEqual(output['logits'].shape, (batch_size, 3))

    def test_generation(self):
        model = QuantumCSTransformer(self.config, task_type='generation')
        
        batch_size = 1
        seq_len = 2
        input_ids = torch.randint(1, self.config.vocab_size, (batch_size, seq_len))
        context_data = {'document_embedding': torch.randn(batch_size, self.config.raw_doc_dim)}

        generated, metrics = model.generate(input_ids, context_data, max_length=5)
        
        self.assertTrue(generated.size(1) > seq_len)
        self.assertEqual(generated.size(0), batch_size)

if __name__ == '__main__':
    unittest.main()
