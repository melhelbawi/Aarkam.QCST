# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum Fragment Encoder - Standalone Implementation
Quantum-aware fragment encoding without classical dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


class QuantumPositionalEncoding(nn.Module):
    """Positional encoding optimized for quantum circuits"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
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
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1)]


class QuantumCharacterCNN(nn.Module):
    """CNN-based character encoder - quantum optimized"""
    
    def __init__(self, char_embed_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(char_embed_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass through CNN layers"""
        x = x.transpose(1, 2)  # [batch_size, char_embed_dim, seq_len]
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.pool(x).squeeze(-1)
        return self.projection(x)


class QuantumAwareAttention(nn.Module):
    """Attention layer with quantum awareness"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
    def forward(self, query, key, value, mask=None):
        """Apply quantum-aware attention"""
        attn_output, attn_weights = self.attention(query, key, value, attn_mask=mask)
        return attn_output, attn_weights


class QuantumMiniTransformer(nn.Module):
    """Lightweight quantum-optimized transformer"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.pos_encoding = QuantumPositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ))
    
    def forward(self, x, mask=None):
        """Forward pass through transformer layers"""
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return x


class QuantumFragmentEncoder(nn.Module):
    """
    Quantum Fragment Encoder - Standalone Implementation
    Encodes text fragments with quantum-aware local context
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get configuration values with fallbacks
        d_model = getattr(config, 'd_model', 512)
        num_heads = getattr(config, 'num_heads', 8)
        
        # Ensure num_heads divides d_model
        if d_model % num_heads != 0:
            num_heads = 8  # Use safe default
            
        char_embed_dim = getattr(config, 'char_embed_dim', 128)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        num_transformer_layers = getattr(config, 'num_transformer_layers', 2)
        transformer_ff_dim = getattr(config, 'transformer_ff_dim', d_model * 4)
        dropout_rate = getattr(config, 'dropout_rate', 0.1)
        
        # Character embedding
        self.char_embedding = nn.Embedding(128, char_embed_dim)
        
        # CNN for local patterns
        self.cnn = QuantumCharacterCNN(
            char_embed_dim,
            hidden_dim,
            d_model
        )
        
        # Mini-transformer for context
        self.transformer = QuantumMiniTransformer(
            d_model,
            num_heads,
            num_transformer_layers,
            transformer_ff_dim
        )
        
        # Quantum feature extraction
        self.quantum_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Ambiguity detection (quantum-enhanced)
        self.ambiguity_score_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, fragments: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode fragments with quantum-aware context
        
        Args:
            fragments: [batch_size, seq_len] tensor of character indices
            context_mask: Optional mask for context
            
        Returns:
            encoded: [batch_size, d_model] encoded fragments
            ambiguity_scores: [batch_size, 1] ambiguity detection scores
        """
        # Embed characters
        char_embedded = self.char_embedding(fragments)  # [batch_size, seq_len, char_embed_dim]
        
        # Extract local patterns with CNN
        cnn_output = self.cnn(char_embedded)  # [batch_size, d_model]
        
        # Encode context with transformer
        transformer_output = self.transformer(char_embedded, mask=context_mask)  # [batch_size, seq_len, d_model]
        context_encoded = transformer_output.mean(dim=1)  # [batch_size, d_model]
        
        # Combine CNN and transformer outputs
        combined = cnn_output + context_encoded
        
        # Quantum projection
        encoded = self.quantum_projection(combined)
        encoded = self.dropout(encoded)
        
        # Detect ambiguity
        ambiguity_scores = self.ambiguity_score_layer(encoded)
        
        return encoded, ambiguity_scores
    
    def encode_batch(self, batch_texts: list, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts"""
        batch_size = len(batch_texts)
        max_len = max(len(text) for text in batch_texts)
        
        # Pad sequences
        fragments = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, text in enumerate(batch_texts):
            text_indices = torch.tensor([ord(c) % 128 for c in text], dtype=torch.long)
            fragments[i, :len(text)] = text_indices
        
        # Create mask for padding
        mask = fragments == 0
        
        return self.forward(fragments, context_mask=mask)
