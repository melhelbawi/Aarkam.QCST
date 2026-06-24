# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum Transformer Utilities - Standalone Implementation
Transformer components without classical dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
import logging

logger = logging.getLogger(__name__)


class QuantumPositionalEncoding(nn.Module):
    """Positional encoding for quantum-enhanced sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class QuantumScaledDotProductAttention(nn.Module):
    """Quantum-aware scaled dot-product attention"""
    
    def __init__(self, d_k: int, dropout: float = 0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """Compute attention with quantum awareness"""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class QuantumMultiHeadAttention(nn.Module):
    """Multi-head attention optimized for quantum integration"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = QuantumScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights


class QuantumFeedForward(nn.Module):
    """Feed-forward network for quantum transformer"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class QuantumTransformerLayer(nn.Module):
    """Single transformer layer with quantum awareness"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = QuantumMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = QuantumFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class QuantumTransformer(nn.Module):
    """
    Quantum-Enhanced Transformer
    Transformer architecture optimized for quantum circuit integration
    """
    
    def __init__(self, d_model: int, num_heads: int, num_layers: int, 
                 d_ff: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.positional_encoding = QuantumPositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            QuantumTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass through quantum transformer
        
        Args:
            x: [batch_size, seq_len, d_model] input tensor
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model] transformed tensor
        """
        # Apply positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class QuantumCSTransformer(nn.Module):
    """
    Quantum-Enhanced CST Transformer
    Transformer specifically designed for Contextual Spectrum Tokenization
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = QuantumPositionalEncoding(config.d_model)
        
        # Transformer backbone
        self.transformer = QuantumTransformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers,
            d_ff=config.transformer_ff_dim,
            dropout=config.dropout_rate
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through quantum CST transformer
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            mask: Optional attention mask
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] output logits
        """
        # Embed tokens
        x = self.token_embedding(token_ids)
        x = self.position_embedding(x)
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x, mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits


def collate_quantum_batch(batch: List[Dict], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Collate function for quantum CST batches
    Handles quantum-specific data organization
    
    Args:
        batch: List of batch items
        device: Device to move tensors to
        
    Returns:
        Dictionary of collated tensors
    """
    token_ids_list = []
    lengths = []
    
    for item in batch:
        if isinstance(item, dict):
            token_ids = item.get('token_ids', item.get('input_ids'))
        else:
            token_ids = item
        
        token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))
        lengths.append(len(token_ids))
    
    # Pad sequences
    max_length = max(lengths)
    padded_token_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_length, dtype=torch.float)
    
    for i, (token_ids, length) in enumerate(zip(token_ids_list, lengths)):
        padded_token_ids[i, :length] = token_ids
        attention_mask[i, :length] = 1.0
    
    return {
        'token_ids': padded_token_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'lengths': torch.tensor(lengths, dtype=torch.long).to(device)
    }
