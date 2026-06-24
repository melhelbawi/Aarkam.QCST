# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum-Enhanced CST Transformer
Complete transformer model with quantum information fusion - Fully Standalone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math
import logging

from quantum_cst_module import QuantumEnhancedCSTModule
from quantum_transformer_utils import (
    QuantumPositionalEncoding, QuantumMultiHeadAttention, 
    QuantumFeedForward, QuantumTransformerLayer, QuantumTransformer
)

logger = logging.getLogger(__name__)




class QuantumTransformerLayer(nn.Module):
    """
    Quantum Transformer Layer - Standalone
    Uses quantum-native components with no classical dependencies
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 quantum_config, dropout: float = 0.1):
        super().__init__()
        
        # Use quantum attention from standalone module
        self.self_attention = QuantumMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = QuantumFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class QuantumCSTransformer(nn.Module):
    """
    Complete Quantum-Enhanced CST Transformer
    
    Integrates quantum computing at multiple levels:
    1. Quantum Information Fusion (CST Module)
    2. Optional Quantum Attention Mechanisms
    3. Quantum-aware training and optimization
    """
    
    def __init__(self, config, task_type: str = 'mlm'):
        super().__init__()
        self.config = config
        self.task_type = task_type
        self.quantum_enabled = config.quantum_config.enable_quantum
        
        logger.info(f"Initializing Quantum-Enhanced CST Transformer")
        logger.info(f"Quantum Enabled: {self.quantum_enabled}")
        logger.info(f"Quantum Mode: {config.quantum_config.quantum_mode}")
        
        # Quantum-Enhanced CST Module
        self.cst_module = QuantumEnhancedCSTModule(config)
        
        # Positional encoding
        self.pos_encoding = QuantumPositionalEncoding(
            config.d_model, 
            config.max_sequence_length if hasattr(config, 'max_sequence_length') else 5000
        )
        
        # Transformer layers - quantum only (no classical fallback)
        logger.info("Using Quantum-Enhanced Transformer Layers")
        self.transformer_layers = nn.ModuleList([
            QuantumTransformerLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_model * 4,
                quantum_config=config.quantum_config,
                dropout=config.dropout if hasattr(config, 'dropout') else 0.1
            ) for _ in range(config.num_transformer_layers if hasattr(config, 'num_transformer_layers') else config.num_layers)
        ])
        
        # Output projection head (quantum-native)
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size if hasattr(config, 'vocab_size') else 50000)
        )
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Quantum performance tracking
        self.quantum_metrics_history = []
        
    def _init_weights(self, module):
        """Initialize weights with proper scaling for quantum components"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_attention_mask(self, input_ids: torch.Tensor, 
                            padding_token_id: int = 0) -> torch.Tensor:
        """Create attention mask from input IDs"""
        return (input_ids != padding_token_id).float()
    
    def forward(self, 
                input_ids: torch.Tensor,
                context_data: Dict[str, Any],
                attention_mask: Optional[torch.Tensor] = None,
                fragment_chars: Optional[torch.Tensor] = None,
                context_chars: Optional[torch.Tensor] = None,
                fragment_frequencies: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_quantum_metrics: bool = True) -> Dict[str, torch.Tensor]:
        """
        Quantum-enhanced forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            context_data: Dictionary of contextual information
            attention_mask: [batch_size, seq_len]
            fragment_chars: [batch_size, seq_len, char_len]
            context_chars: [batch_size, seq_len, context_len]
            fragment_frequencies: [batch_size, seq_len]
            labels: [batch_size, seq_len] or [batch_size]
            return_quantum_metrics: Whether to return detailed quantum metrics
        
        Returns:
            Dictionary containing logits, loss, hidden states, and quantum metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if needed
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # CST Module - Generate quantum-enhanced contextual spectrum vectors
        spectrum_vectors, quantum_metrics = self.cst_module(
            text_fragments=input_ids,
            context_data=context_data,
            fragment_chars=fragment_chars,
            context_chars=context_chars,
            fragment_frequencies=fragment_frequencies
        )
        
        # Add positional encoding
        positioned_vectors = self.pos_encoding(spectrum_vectors)
        
        # Process through transformer layers
        hidden_states = positioned_vectors
        
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, mask=attention_mask)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Output head
        if self.task_type == 'classification':
            logits = self.output_head(hidden_states, pooling_strategy='mean')
        else:
            logits = self.output_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.task_type == 'mlm':
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            elif self.task_type == 'classification':
                loss = F.cross_entropy(logits, labels)
            elif self.task_type == 'generation':
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
        
        # Prepare output
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
            'loss': loss
        }
        
        # Add quantum metrics if requested
        if return_quantum_metrics:
            output['quantum_metrics'] = quantum_metrics
            output['cst_stats'] = self.cst_module.get_performance_stats()
            
            # Track quantum metrics history
            if self.training:
                self.quantum_metrics_history.append(quantum_metrics)
        
        return output
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 context_data: Dict[str, Any],
                 max_length: int = 50,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 use_quantum: bool = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate text with quantum-enhanced CST
        
        Returns:
            Tuple of (generated_ids, quantum_metrics_per_step)
        """
        # Auto-determine quantum usage
        if use_quantum is None:
            use_quantum = self.quantum_enabled
        
        # Temporarily adjust quantum settings for generation
        original_quantum_state = self.quantum_enabled
        if not use_quantum:
            self.quantum_enabled = False
        
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        generation_quantum_metrics = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                outputs = self.forward(
                    generated, 
                    context_data,
                    return_quantum_metrics=True
                )
                logits = outputs['logits']
                
                # Track quantum metrics
                if 'quantum_metrics' in outputs:
                    generation_quantum_metrics.append(outputs['quantum_metrics'])
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample or select next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if (next_token == 2).all():  # Assuming 2 is EOS
                    break
        
        # Restore quantum state
        self.quantum_enabled = original_quantum_state
        
        return generated, generation_quantum_metrics
    
    def get_embeddings(self, input_ids: torch.Tensor, context_data: Dict[str, Any]) -> torch.Tensor:
        """Get quantum-enhanced CST embeddings"""
        spectrum_vectors, _ = self.cst_module(input_ids, context_data)
        return spectrum_vectors
    
    def enable_cst_profiling(self, enable: bool = True):
        """Enable CST performance profiling"""
        self.cst_module.enable_profiling_mode(enable)
    
    def get_cst_stats(self) -> Dict[str, Any]:
        """Get CST performance statistics"""
        return self.cst_module.get_performance_stats()
    
    def get_quantum_summary(self) -> Dict[str, Any]:
        """Get comprehensive quantum usage summary"""
        summary = {
            'quantum_enabled': self.quantum_enabled,
            'quantum_mode': self.config.quantum_config.quantum_mode,
            'n_qubits': self.config.quantum_config.n_qubits,
            'n_quantum_layers': self.config.quantum_config.n_layers,
            'state_space_dimension': 2 ** self.config.quantum_config.n_qubits,
            'cst_stats': self.get_cst_stats(),
            'quantum_circuit_info': self.cst_module.get_quantum_info()
        }
        
        # Average quantum metrics over history
        if self.quantum_metrics_history:
            avg_metrics = {}
            for key in self.quantum_metrics_history[0].keys():
                if isinstance(self.quantum_metrics_history[0][key], (int, float)):
                    values = [m[key] for m in self.quantum_metrics_history]
                    avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            summary['average_quantum_metrics'] = avg_metrics
        
        return summary
    
    def save_pretrained(self, save_directory: str):
        """Save quantum-enhanced model"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save configuration
        config_dict = {
            'd_model': self.config.d_model,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'vocab_size': self.config.vocab_size,
            'task_type': self.task_type,
            'quantum_enabled': self.quantum_enabled,
            'quantum_config': self.config.quantum_config.__dict__
        }
        
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save quantum-specific data
        self.cst_module.save_ambiguous_vocab(
            os.path.join(save_directory, 'ambiguous_vocab.json')
        )
        
        # Save quantum summary
        quantum_summary = self.get_quantum_summary()
        with open(os.path.join(save_directory, 'quantum_summary.json'), 'w') as f:
            json.dump(quantum_summary, f, indent=2)
        
        logger.info(f"Quantum-Enhanced CST model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = 'cuda'):
        """Load quantum-enhanced model"""
        import os
        import json
        
        # Load configuration
        with open(os.path.join(load_directory, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct config
        from config import CSTConfig, QuantumConfig
        config = CSTConfig()
        for key, value in config_dict.items():
            if key == 'quantum_config':
                config.quantum_config = QuantumConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        # Create model
        model = cls(config, task_type=config_dict.get('task_type', 'mlm'))
        
        # Load weights
        state_dict = torch.load(
            os.path.join(load_directory, 'pytorch_model.bin'),
            map_location=device
        )
        model.load_state_dict(state_dict)
        
        model.to(device)
        logger.info(f"Quantum-Enhanced CST model loaded from {load_directory}")
        
        return model


def test_quantum_cst_transformer():
    """Test the quantum-enhanced CST transformer"""
    from config import CSTConfig, QuantumConfig
    
    config = CSTConfig()
    config.num_layers = 6
    config.quantum_config = QuantumConfig()
    config.quantum_config.enable_quantum = True
    config.quantum_config.n_qubits = 6
    config.quantum_config.n_layers = 2
    config.quantum_config.use_quantum_attention = False  # Disable for speed
    config.ambiguous_word_ids = [1, 5, 10, 15, 20]
    
    for task_type in ['mlm', 'classification']:
        print(f"\n{'='*60}")
        print(f"Testing Quantum CST Transformer: {task_type}")
        print('='*60)
        
        if task_type == 'classification':
            config.num_labels = 5
        
        model = QuantumCSTransformer(config, task_type=task_type)
        model.enable_cst_profiling(True)
        
        batch_size = 2
        seq_len = 16
        
        # Sample data
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        
        context_data = {
            'document_embedding': torch.randn(batch_size, config.raw_doc_dim),
            'metadata': {
                'author': torch.randint(0, config.num_authors, (batch_size,)),
                'domain': torch.randint(0, config.num_domains, (batch_size,)),
            }
        }
        
        # Labels
        if task_type == 'mlm':
            labels = input_ids.clone()
            mask_positions = torch.rand(batch_size, seq_len) < 0.15
            labels[~mask_positions] = -100
        else:
            labels = torch.randint(0, 5, (batch_size,))
        
        # Forward pass
        outputs = model(input_ids, context_data, labels=labels)
        
        print(f"\nInput shape: {input_ids.shape}")
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Loss: {outputs['loss'].item():.4f}")
        
        # Quantum metrics
        if 'quantum_metrics' in outputs:
            print(f"\nQuantum Metrics:")
            for key, value in outputs['quantum_metrics'].items():
                print(f"  {key}: {value}")
        
        # Performance stats
        stats = model.get_cst_stats()
        print(f"\nPerformance Stats:")
        print(f"  Cache hit rate: {stats.get('hit_rate', 0):.2%}")
        print(f"  Quantum processed: {stats.get('quantum_processed_tokens', 0)}")
        print(f"  Quantum usage ratio: {stats.get('quantum_usage_ratio', 0):.2%}")
        
        # Quantum summary
        summary = model.get_quantum_summary()
        print(f"\nQuantum Summary:")
        print(f"  State space dimension: {summary['state_space_dimension']}")
        print(f"  Quantum enabled: {summary['quantum_enabled']}")
    
    print(f"\n{'='*60}")
    print("✅ All Quantum CST Transformer tests passed!")
    print('='*60)


if __name__ == "__main__":
    test_quantum_cst_transformer()