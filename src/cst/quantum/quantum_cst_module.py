# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum-Enhanced CST Module
Integrates quantum computing into the core CST architecture
Fully standalone - no classical dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
from collections import OrderedDict
import time
import logging

from quantum_information_fuser import QuantumInformationFuser
from quantum_fragment_encoder import QuantumFragmentEncoder
from quantum_cst_config import QuantumConfig

logger = logging.getLogger(__name__)


class LRUCache:
    """Enhanced LRU cache with quantum state caching support"""
    
    def __init__(self, capacity: int, enable_quantum_cache: bool = True):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.quantum_cache = OrderedDict() if enable_quantum_cache else None
        self.hits = 0
        self.misses = 0
        self.quantum_hits = 0
        self.quantum_misses = 0
        
    def get(self, key: str, is_quantum: bool = False) -> Optional[torch.Tensor]:
        cache_dict = self.quantum_cache if (is_quantum and self.quantum_cache is not None) else self.cache
        
        if key in cache_dict:
            cache_dict.move_to_end(key)
            if is_quantum:
                self.quantum_hits += 1
            else:
                self.hits += 1
            return cache_dict[key].clone()
        else:
            if is_quantum:
                self.quantum_misses += 1
            else:
                self.misses += 1
            return None
    
    def put(self, key: str, value: torch.Tensor, is_quantum: bool = False):
        cache_dict = self.quantum_cache if (is_quantum and self.quantum_cache is not None) else self.cache
        
        if key in cache_dict:
            cache_dict.move_to_end(key)
        else:
            if len(cache_dict) >= self.capacity:
                cache_dict.popitem(last=False)
        
        cache_dict[key] = value.clone().detach()
    
    def clear(self):
        self.cache.clear()
        if self.quantum_cache is not None:
            self.quantum_cache.clear()
        self.hits = 0
        self.misses = 0
        self.quantum_hits = 0
        self.quantum_misses = 0
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        quantum_total = self.quantum_hits + self.quantum_misses
        quantum_hit_rate = self.quantum_hits / quantum_total if quantum_total > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'capacity': self.capacity,
            'quantum_hits': self.quantum_hits,
            'quantum_misses': self.quantum_misses,
            'quantum_hit_rate': quantum_hit_rate,
            'quantum_cache_size': len(self.quantum_cache) if self.quantum_cache else 0
        }


class QuantumAmbiguityClassifier(nn.Module):
    """Quantum-enhanced ambiguity classification (optional)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_quantum = config.quantum_config.quantum_ambiguity_classifier
        
        # Classical ambiguity detection
        self.register_buffer(
            'ambiguous_vocab', 
            torch.tensor(config.ambiguous_word_ids if config.ambiguous_word_ids else [])
        )
        
        # Context-based classifier
        context_input_dim = config.fragment_encoding_dim + config.context_feature_dim
        self.context_classifier = nn.Sequential(
            nn.Linear(context_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Quantum enhancement (if enabled)
        if self.use_quantum:
            from quantum_information_fuser import HybridQuantumClassical
            self.quantum_classifier = HybridQuantumClassical(
                input_dim=context_input_dim,
                output_dim=1,
                quantum_config=config.quantum_config
            )
        
        self.frequency_classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.combination_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        self.ambiguity_threshold = config.ambiguity_threshold
        
    def forward(self, 
                fragment_ids: torch.Tensor, 
                context_features: torch.Tensor,
                fragment_frequencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = fragment_ids.size(0)
        ambiguity_scores = torch.zeros(batch_size, device=fragment_ids.device)
        
        # 1. Vocabulary-based
        if len(self.ambiguous_vocab) > 0:
            vocab_ambiguous = torch.isin(fragment_ids, self.ambiguous_vocab).float()
            ambiguity_scores += self.combination_weights[0] * vocab_ambiguous
        
        # 2. Context-based (quantum or classical)
        if context_features.size(1) >= self.config.context_feature_dim:
            fragment_encoding = torch.zeros(batch_size, self.config.fragment_encoding_dim, 
                                          device=fragment_ids.device)
            combined_features = torch.cat([fragment_encoding, 
                                         context_features[:, :self.config.context_feature_dim]], dim=1)
            
            if self.use_quantum:
                context_scores = self.quantum_classifier(combined_features).squeeze(-1)
            else:
                context_scores = self.context_classifier(combined_features).squeeze(-1)
            
            ambiguity_scores += self.combination_weights[1] * context_scores
        
        # 3. Frequency-based
        if fragment_frequencies is not None:
            freq_scores = self.frequency_classifier(fragment_frequencies.unsqueeze(-1)).squeeze(-1)
            ambiguity_scores += self.combination_weights[2] * freq_scores
        
        return ambiguity_scores > self.ambiguity_threshold


class ProjectionHead(nn.Module):
    """Projects fused representation to transformer embedding dimension"""
    
    def __init__(self, config):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(config.fused_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        
        self.use_residual = config.fused_dim == config.d_model
        if not self.use_residual and hasattr(config, 'enable_projection_residual'):
            self.residual_proj = nn.Linear(config.fused_dim, config.d_model)
            self.use_residual = config.enable_projection_residual
        
    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        output = self.projection(fused_representation)
        
        if self.use_residual:
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(fused_representation)
            else:
                residual = fused_representation
            output = output + residual
            
        return output


class QuantumEnhancedCSTModule(nn.Module):
    """
    Quantum-Enhanced Contextual Spectrum Tokenization Module
    
    Integrates quantum computing for enhanced information fusion
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantum_enabled = config.quantum_config.enable_quantum
        
        # Core components - fully quantum standalone
        self.fragment_encoder = QuantumFragmentEncoder(config)
        
        # Information Fuser - Quantum only (no classical fallback)
        logger.info("Initializing Quantum Information Fuser (Standalone)")
        self.information_fuser = QuantumInformationFuser(config, config.quantum_config)
        
        self.projection_head = ProjectionHead(config)
        self.ambiguity_classifier = QuantumAmbiguityClassifier(config)
        
        # Static embeddings fallback
        self.static_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.static_embeddings.weight, mean=0.0, std=0.02)
        
        # Enhanced caching with quantum support
        self.cache = LRUCache(
            config.cache_size, 
            enable_quantum_cache=self.quantum_enabled
        )
        
        # Performance tracking
        self.enable_profiling = False
        self.profile_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'ambiguous_tokens': 0,
            'static_tokens': 0,
            'quantum_processed_tokens': 0,
            'classical_processed_tokens': 0,
            'total_forward_time': 0.0,
            'quantum_forward_time': 0.0,
            'classical_forward_time': 0.0,
            'num_forward_calls': 0
        }
        
    def _compute_cache_key(self, fragment_data: Dict[str, Any], 
                          context_data: Dict[str, Any],
                          use_quantum: bool = False) -> str:
        """Compute cache key with quantum indicator"""
        key_components = {
            'fragment_id': fragment_data.get('fragment_id', '').item() if torch.is_tensor(fragment_data.get('fragment_id')) else str(fragment_data.get('fragment_id', '')),
            'context_hash': self._hash_context(context_data),
            'quantum': use_quantum
        }
        
        key_string = json.dumps(key_components, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _hash_context(self, context_data: Dict[str, Any]) -> str:
        """Create a hash of context data for caching"""
        context_summary = {}
        
        for key, value in context_data.items():
            if isinstance(value, torch.Tensor):
                context_summary[key] = {
                    'shape': list(value.shape),
                    'mean': float(value.mean().item()) if value.numel() > 0 else 0.0,
                    'std': float(value.std().item()) if value.numel() > 0 else 0.0
                }
            elif isinstance(value, dict):
                context_summary[key] = self._hash_context(value)
            else:
                context_summary[key] = str(value)
        
        return hashlib.md5(json.dumps(context_summary, sort_keys=True).encode()).hexdigest()[:16]
    
    def _compute_dynamic_embedding(self, fragment_data: Dict[str, Any], 
                                  context_data: Dict[str, Any]) -> Tuple[torch.Tensor, bool]:
        """
        Compute dynamic embedding using quantum-enhanced or classical pipeline
        
        Returns:
            Tuple of (embedding, used_quantum)
        """
        use_quantum = (self.quantum_enabled and 
                      config.quantum_config.quantum_information_fuser and
                      self.training)  # Use quantum mainly during training
        
        # Time tracking
        start_time = time.time() if self.enable_profiling else 0
        
        # Extract fragment encoding
        fragment_encoding = self.fragment_encoder(
            fragment_data['fragment_chars'],
            fragment_data['context_chars'], 
            fragment_data.get('fragment_positions')
        )
        
        # Fuse with contextual information (quantum or classical)
        fused_representation = self.information_fuser(fragment_encoding, context_data)
        
        # Project to output space
        output_embedding = self.projection_head(fused_representation)
        
        # Track timing
        if self.enable_profiling:
            elapsed = time.time() - start_time
            if use_quantum:
                self.profile_stats['quantum_forward_time'] += elapsed
            else:
                self.profile_stats['classical_forward_time'] += elapsed
        
        return output_embedding, use_quantum
    
    def forward(self, 
                text_fragments: torch.Tensor, 
                context_data: Dict[str, Any],
                fragment_chars: Optional[torch.Tensor] = None,
                context_chars: Optional[torch.Tensor] = None,
                fragment_frequencies: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantum-enhanced forward pass
        
        Returns:
            Tuple of (output_vectors, quantum_metrics)
        """
        start_time = time.time() if self.enable_profiling else 0
        
        batch_size, seq_len = text_fragments.shape
        device = text_fragments.device
        
        # Initialize output
        output_vectors = torch.zeros(batch_size, seq_len, self.config.d_model, device=device)
        
        # Track quantum usage
        quantum_tokens_processed = 0
        classical_tokens_processed = 0
        
        for i in range(seq_len):
            fragment_ids = text_fragments[:, i]
            
            # Prepare fragment data
            fragment_data = {
                'fragment_id': fragment_ids,
                'fragment_chars': fragment_chars[:, i] if fragment_chars is not None else None,
                'context_chars': context_chars[:, i] if context_chars is not None else None,
                'fragment_positions': torch.full((batch_size,), i, device=device)
            }
            
            # Prepare context features
            context_features = torch.zeros(batch_size, self.config.context_feature_dim, device=device)
            if 'document_embedding' in context_data:
                doc_emb = context_data['document_embedding']
                feature_dim = min(self.config.context_feature_dim, doc_emb.size(-1))
                context_features[:, :feature_dim] = doc_emb[:, :feature_dim]
            
            # Determine ambiguity
            freqs = fragment_frequencies[:, i] if fragment_frequencies is not None else None
            is_ambiguous = self.ambiguity_classifier(fragment_ids, context_features, freqs)
            
            # Process each sample
            for b in range(batch_size):
                if is_ambiguous[b]:
                    # Try cache first
                    sample_fragment_data = {k: v[b] if v is not None else None 
                                          for k, v in fragment_data.items()}
                    sample_context_data = {k: v[b] if isinstance(v, torch.Tensor) else v 
                                         for k, v in context_data.items()}
                    
                    use_quantum = (self.quantum_enabled and 
                                 self.config.quantum_config.quantum_information_fuser)
                    cache_key = self._compute_cache_key(sample_fragment_data, 
                                                       sample_context_data,
                                                       use_quantum)
                    cached_vector = self.cache.get(cache_key, is_quantum=use_quantum)
                    
                    if cached_vector is not None:
                        output_vectors[b, i] = cached_vector
                        if self.enable_profiling:
                            self.profile_stats['cache_hits'] += 1
                    else:
                        # Compute dynamic embedding
                        dynamic_vector, used_quantum = self._compute_dynamic_embedding(
                            sample_fragment_data, sample_context_data
                        )
                        output_vectors[b, i] = dynamic_vector.squeeze(0) if dynamic_vector.dim() > 1 else dynamic_vector
                        
                        # Cache the result
                        self.cache.put(cache_key, output_vectors[b, i], is_quantum=used_quantum)
                        
                        if self.enable_profiling:
                            self.profile_stats['cache_misses'] += 1
                            self.profile_stats['ambiguous_tokens'] += 1
                            if used_quantum:
                                quantum_tokens_processed += 1
                            else:
                                classical_tokens_processed += 1
                else:
                    # Use static embedding
                    output_vectors[b, i] = self.static_embeddings(fragment_ids[b])
                    if self.enable_profiling:
                        self.profile_stats['static_tokens'] += 1
        
        # Update statistics
        if self.enable_profiling:
            self.profile_stats['total_forward_time'] += time.time() - start_time
            self.profile_stats['num_forward_calls'] += 1
            self.profile_stats['quantum_processed_tokens'] += quantum_tokens_processed
            self.profile_stats['classical_processed_tokens'] += classical_tokens_processed
        
        # Quantum metrics
        quantum_metrics = {
            'quantum_tokens_in_batch': quantum_tokens_processed,
            'classical_tokens_in_batch': classical_tokens_processed,
            'quantum_ratio': quantum_tokens_processed / (quantum_tokens_processed + classical_tokens_processed + 1e-10)
        }
        
        if hasattr(self.information_fuser, 'get_quantum_circuit_info'):
            quantum_metrics.update(self.information_fuser.get_quantum_circuit_info())
        
        return output_vectors, quantum_metrics
    
    def enable_profiling_mode(self, enable: bool = True):
        """Enable or disable performance profiling"""
        self.enable_profiling = enable
        if enable:
            self.profile_stats = {k: 0 if isinstance(v, (int, float)) else v 
                                for k, v in self.profile_stats.items()}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.profile_stats.copy()
        cache_stats = self.cache.stats()
        stats.update(cache_stats)
        
        # Add derived metrics
        if stats['num_forward_calls'] > 0:
            stats['avg_forward_time'] = stats['total_forward_time'] / stats['num_forward_calls']
            stats['avg_quantum_time'] = stats['quantum_forward_time'] / stats['num_forward_calls']
            stats['avg_classical_time'] = stats['classical_forward_time'] / stats['num_forward_calls']
        
        total_tokens = (stats['ambiguous_tokens'] + stats['static_tokens'])
        if total_tokens > 0:
            stats['ambiguous_ratio'] = stats['ambiguous_tokens'] / total_tokens
            stats['static_ratio'] = stats['static_tokens'] / total_tokens
        
        if stats['ambiguous_tokens'] > 0:
            quantum_processed = stats['quantum_processed_tokens']
            classical_processed = stats['classical_processed_tokens']
            total_processed = quantum_processed + classical_processed
            
            if total_processed > 0:
                stats['quantum_usage_ratio'] = quantum_processed / total_processed
        
        return stats
    
    def get_quantum_info(self) -> Dict[str, Any]:
        """Get quantum-specific information"""
        if hasattr(self.information_fuser, 'get_quantum_circuit_info'):
            return self.information_fuser.get_quantum_circuit_info()
        return {'quantum_enabled': False}
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()


# Alias for backward compatibility
CSTModule = QuantumEnhancedCSTModule


def test_quantum_cst_module():
    """Test the quantum-enhanced CST module"""
    from config import CSTConfig, QuantumConfig
    
    config = CSTConfig()
    config.quantum_config = QuantumConfig()
    config.quantum_config.enable_quantum = True
    config.quantum_config.n_qubits = 6
    config.quantum_config.n_layers = 2
    config.ambiguous_word_ids = [1, 5, 10, 15, 20]
    
    cst = QuantumEnhancedCSTModule(config)
    cst.enable_profiling_mode(True)
    
    batch_size = 2
    seq_len = 8
    
    # Sample input
    text_fragments = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    fragment_chars = torch.randint(0, config.char_vocab_size, (batch_size, seq_len, 32))
    context_chars = torch.randint(0, config.char_vocab_size, (batch_size, seq_len, 64))
    
    context_data = {
        'document_embedding': torch.randn(batch_size, config.raw_doc_dim),
        'metadata': {
            'author': torch.randint(0, config.num_authors, (batch_size,)),
            'domain': torch.randint(0, config.num_domains, (batch_size,)),
        }
    }
    
    # Forward pass
    output, quantum_metrics = cst(text_fragments, context_data, fragment_chars, context_chars)
    
    print(f"Input shape: {text_fragments.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nQuantum Metrics:")
    for key, value in quantum_metrics.items():
        print(f"  {key}: {value}")
    
    # Performance stats
    stats = cst.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in list(stats.items())[:10]:
        print(f"  {key}: {value}")
    
    # Quantum info
    quantum_info = cst.get_quantum_info()
    print("\nQuantum Circuit Info:")
    for key, value in quantum_info.items():
        if key != 'fragment_circuit':
            print(f"  {key}: {value}")
    
    print("\n✅ Quantum-Enhanced CST Module test passed!")


if __name__ == "__main__":
    test_quantum_cst_module()