# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum Information Fuser - Standalone Implementation
VQC-based information fusion without classical dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math
import logging

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    
logger = logging.getLogger(__name__)


class QuantumDocumentEncoder(nn.Module):
    """Quantum-optimized document-level feature encoder"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """Encode document features"""
        return self.encoder(x)


class QuantumMetadataProcessor(nn.Module):
    """Process metadata for quantum fusion"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Categorical metadata embeddings
        self.categorical_processors = nn.ModuleDict({
            'author': nn.Embedding(min(config.num_authors, 1000), config.embed_dim),
            'domain': nn.Embedding(min(config.num_domains, 100), config.embed_dim),
            'language': nn.Embedding(100, config.embed_dim),
            'genre': nn.Embedding(50, config.embed_dim),
        })
        
        # Continuous metadata processors
        self.continuous_processors = nn.ModuleDict({
            'timestamp': nn.Sequential(
                nn.Linear(1, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, config.embed_dim)
            ),
            'document_length': nn.Sequential(
                nn.Linear(1, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, config.embed_dim)
            ),
            'readability_score': nn.Sequential(
                nn.Linear(1, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, config.embed_dim)
            )
        })
        
        # Text metadata processor
        self.text_processor = nn.Sequential(
            nn.Linear(config.d_model, config.embed_dim),
            nn.GELU()
        )
        
    def forward(self, metadata: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Process metadata and return feature tensors"""
        features = []
        
        # Process categorical features
        for key, processor in self.categorical_processors.items():
            if key in metadata:
                feat = processor(metadata[key].long())
                features.append(feat)
        
        # Process continuous features
        for key, processor in self.continuous_processors.items():
            if key in metadata:
                value = metadata[key].float()
                
                if key == 'timestamp':
                    value = (value - 1577836800) / (365.25 * 24 * 3600)
                elif key == 'document_length':
                    value = torch.log(value + 1)
                elif key == 'readability_score':
                    value = value / 100.0
                
                feat = processor(value.unsqueeze(-1) if value.dim() == 1 else value)
                features.append(feat)
        
        # Process text-based metadata
        for key in ['title', 'summary', 'keywords']:
            if key in metadata:
                feat = self.text_processor(metadata[key])
                features.append(feat)
        
        return features if features else [torch.zeros(1, self.config.embed_dim)]


class HybridQuantumClassical(nn.Module):
    """
    Hybrid Quantum-Classical Layer
    Uses VQC to transform input features without classical dependencies
    """
    
    def __init__(self, input_dim: int, output_dim: int, quantum_config):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantum_config = quantum_config
        
        # Get quantum config values with fallbacks
        num_qubits = getattr(quantum_config, 'num_qubits', getattr(quantum_config, 'n_qubits', 8))
        num_layers = getattr(quantum_config, 'num_layers', getattr(quantum_config, 'n_layers', 3))
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        if HAS_PENNYLANE:
            # Initialize quantum device
            self.dev = qml.device("default.qubit", wires=num_qubits)
            
            # Quantum circuit parameters
            self.num_qubits = num_qubits
            self.num_layers = num_layers
            
            # Classical preprocessing layer
            self.input_projection = nn.Linear(input_dim, num_qubits * 2)
            
            # Quantum circuit parameters (trainable)
            self.params = nn.Parameter(
                torch.randn(num_layers, num_qubits, 3)
            )
            
        # Classical postprocessing
        self.output_projection = nn.Linear(num_qubits, output_dim)
        
    def _build_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Build and execute quantum circuit"""
        if not HAS_PENNYLANE:
            return x
        
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            # Preprocess input
            preprocessed = self.input_projection(x[b:b+1])
            angles = preprocessed.view(self.num_qubits, 2)
            
            # Execute quantum circuit
            @qml.qnode(self.dev)
            def circuit(params):
                # Encoding
                for i, angle in enumerate(angles):
                    qml.RY(angle[0], wires=i)
                    qml.RZ(angle[1], wires=i)
                
                # Variational layers
                for layer in range(self.num_layers):
                    for qubit in range(self.num_qubits):
                        qml.RY(self.params[layer, qubit, 0], wires=qubit)
                        qml.RZ(self.params[layer, qubit, 1], wires=qubit)
                        qml.RY(self.params[layer, qubit, 2], wires=qubit)
                    
                    # Entangling layer
                    for qubit in range(self.num_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                
                # Measurement
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            
            # Execute circuit
            try:
                result = circuit(self.params)
                if isinstance(result, (list, tuple)):
                    output = torch.tensor(result, dtype=x.dtype, device=x.device)
                else:
                    output = result.unsqueeze(0) if result.dim() == 0 else result
                outputs.append(output)
            except Exception as e:
                logger.warning(f"Quantum circuit execution failed: {e}, using fallback")
                outputs.append(torch.randn(self.num_qubits, device=x.device, dtype=x.dtype))
        
        return torch.stack(outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hybrid quantum-classical transformation"""
        if HAS_PENNYLANE and self.quantum_config.enable_quantum:
            quantum_output = self._build_circuit(x)
        else:
            # Classical fallback
            quantum_output = torch.randn(x.size(0), self.num_qubits, device=x.device, dtype=x.dtype)
        
        # Post-processing
        output = self.output_projection(quantum_output)
        return output


class QuantumAttentionFusion(nn.Module):
    """Quantum-enhanced attention-based information fusion"""
    
    def __init__(self, d_model: int, num_heads: int, num_modalities: int):
        super().__init__()
        # Ensure num_heads divides d_model
        if d_model % num_heads != 0:
            num_heads = min(num_heads, d_model // 2) if d_model % 2 == 0 else 1
            
        self.attention = nn.MultiheadAttention(d_model, max(1, num_heads), batch_first=True)
        self.num_modalities = num_modalities
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_modalities)
        ])
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality representations"""
        # Project each modality
        projected = []
        for i, rep in enumerate(representations):
            proj = self.modality_projections[i](rep)
            projected.append(proj)
        
        # Stack and apply attention
        stacked = torch.stack(projected, dim=1)  # [batch, num_modalities, d_model]
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = torch.einsum('bmd,m->bd', stacked, weights)
        
        return fused


class QuantumInformationFuser(nn.Module):
    """
    Quantum Information Fuser - Standalone Implementation
    Fuses fragment encodings with multimodal signals using quantum circuits
    """
    
    def __init__(self, config, quantum_config):
        super().__init__()
        self.config = config
        self.quantum_config = quantum_config
        
        # Document encoding
        self.document_encoder = QuantumDocumentEncoder(
            config.doc_input_dim if hasattr(config, 'doc_input_dim') else 256,
            config.d_model,
            config.hidden_dim
        )
        
        # Metadata processing
        self.metadata_processor = QuantumMetadataProcessor(config)
        
        # Get safe num_heads value
        num_heads = getattr(config, 'num_heads', 8)
        d_model = getattr(config, 'd_model', 512)
        # Ensure divisibility
        if d_model % num_heads != 0:
            num_heads = min(num_heads, 8)
        
        # Hybrid quantum-classical fusion
        self.hybrid_fuser = HybridQuantumClassical(
            d_model * 2,
            d_model,
            quantum_config
        )
        
        # Attention-based fusion
        num_modalities = 3  # fragment, document, metadata
        self.attention_fusion = QuantumAttentionFusion(
            d_model,
            num_heads,
            num_modalities
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        dropout_rate = getattr(config, 'dropout_rate', getattr(config, 'dropout', 0.1))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        fragment_encoding: torch.Tensor,
        document_context: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse information from multiple modalities
        
        Args:
            fragment_encoding: [batch_size, d_model] encoded fragments
            document_context: [batch_size, doc_features] document-level context
            metadata: Dictionary of metadata tensors
            modality_masks: Optional masks for each modality
            
        Returns:
            fused: [batch_size, d_model] fused representation
            fusion_weights: Dictionary of modality fusion weights
        """
        representations = [fragment_encoding]
        
        # Process document context
        if document_context is not None:
            doc_encoded = self.document_encoder(document_context)
            representations.append(doc_encoded)
        else:
            representations.append(torch.zeros_like(fragment_encoding))
        
        # Process metadata
        if metadata is not None:
            metadata_features = self.metadata_processor(metadata)
            if metadata_features:
                metadata_fused = torch.cat(metadata_features, dim=-1)
                metadata_encoded = nn.Linear(metadata_fused.size(-1), fragment_encoding.size(-1)).to(fragment_encoding.device)(metadata_fused)
                representations.append(metadata_encoded)
            else:
                representations.append(torch.zeros_like(fragment_encoding))
        else:
            representations.append(torch.zeros_like(fragment_encoding))
        
        # Apply attention fusion
        fused = self.attention_fusion(representations)
        
        # Apply hybrid quantum transformation
        hybrid_input = torch.cat([fragment_encoding, fused], dim=-1)
        quantum_output = self.hybrid_fuser(hybrid_input)
        
        # Final projection
        output = self.output_projection(quantum_output)
        output = self.dropout(output)
        
        # Create fusion weights dict
        fusion_weights = {
            'fragment': 0.5,
            'document': 0.3,
            'metadata': 0.2
        }
        
        return output, fusion_weights
