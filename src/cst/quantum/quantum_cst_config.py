# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum-Enhanced CST Configuration
Integrated configuration for classical and quantum components
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml
import os


@dataclass
class QuantumConfig:
    """Quantum computing configuration for CST"""
    
    # Enable/Disable Quantum Features
    enable_quantum: bool = True
    quantum_mode: str = 'hybrid'  # 'hybrid', 'quantum_only', 'classical_only'
    
    # Quantum Circuit Architecture
    n_qubits: int = 8
    n_layers: int = 3
    rotation_gates: List[str] = field(default_factory=lambda: ['RX', 'RY', 'RZ'])
    entanglement_pattern: str = 'full'  # 'linear', 'circular', 'full', 'custom'
    
    # Quantum Feature Encoding
    encoding_type: str = 'amplitude'  # 'amplitude', 'angle', 'iqp'
    feature_scaling: str = 'normalize'
    max_features_per_qubit: int = 2
    
    # Hybrid Quantum-Classical Integration
    quantum_weight: float = 0.5
    use_adaptive_weight: bool = True
    gradient_method: str = 'parameter_shift'  # 'parameter_shift', 'backprop', 'finite_diff'
    
    # Quantum Backend Configuration
    backend: str = 'default.qubit'  # 'default.qubit', 'qiskit.aer', 'qiskit.ibm'
    shots: Optional[int] = None  # None for exact simulation, int for sampling
    diff_method: str = 'best'
    
    # Advanced Quantum Features
    use_data_reuploading: bool = True
    use_quantum_pooling: bool = True
    use_quantum_attention: bool = True
    noise_model: Optional[str] = None  # 'depolarizing', 'amplitude_damping', None
    
    # Quantum Optimization
    quantum_lr: float = 0.01
    quantum_optimizer: str = 'adam'  # 'adam', 'sgd', 'spsa', 'rotosolve'
    use_parameter_shift_rules: bool = True
    
    # Measurement Strategy
    measurement_type: str = 'pauli_xyz'  # 'pauli_z', 'pauli_xyz', 'state_vector'
    use_stochastic_measurement: bool = True
    
    # Quantum Components to Use
    quantum_fragment_encoder: bool = True
    quantum_information_fuser: bool = True
    quantum_ambiguity_classifier: bool = False
    quantum_attention_mechanism: bool = True
    
    # Performance Monitoring
    track_quantum_metrics: bool = True
    log_circuit_depth: bool = True
    log_gate_count: bool = True


@dataclass
class CSTConfig:
    """Main configuration class for Quantum-Enhanced CST model"""
    
    # Model Architecture
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 30522
    max_sequence_length: int = 512
    dropout: float = 0.1
    
    # CST Module Configuration
    char_vocab_size: int = 256
    char_embed_dim: int = 64
    context_window: int = 64
    fragment_encoding_dim: int = 256
    document_encoding_dim: int = 128
    metadata_dim: int = 64
    multimodal_dim: int = 512
    hidden_dim: int = 512
    fused_dim: int = 384
    embed_dim: int = 64
    
    # External Dimensions
    raw_doc_dim: int = 1024
    clip_dim: int = 512
    audio_dim: int = 768
    context_feature_dim: int = 128
    
    # Ambiguity Classification
    ambiguous_word_ids: List[int] = field(default_factory=list)
    ambiguity_threshold: float = 0.5
    
    # Caching
    cache_size: int = 10000
    l1_cache_size: int = 5000
    
    # Training Configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 10000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Contrastive Learning
    temperature: float = 0.07
    contrastive_weight: float = 0.3
    mlm_weight: float = 0.7
    
    # Regularization
    reference_update_freq: int = 1000
    reference_momentum: float = 0.999
    drift_regularization_weight: float = 0.1
    
    # Quantum Configuration
    quantum_config: QuantumConfig = field(default_factory=QuantumConfig)
    
    # Quantum-Specific Training
    quantum_loss_weight: float = 0.2  # Weight for quantum-specific losses
    use_quantum_contrastive: bool = True
    quantum_entanglement_regularization: float = 0.05
    
    # Production Settings
    max_batch_size: int = 64
    inference_cache_ttl: int = 3600
    
    # Metadata Processing
    num_authors: int = 10000
    num_domains: int = 100
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'CSTConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested quantum config
        if 'quantum_config' in config_dict:
            quantum_dict = config_dict.pop('quantum_config')
            quantum_config = QuantumConfig(**quantum_dict)
            config_dict['quantum_config'] = quantum_config
        
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        config_dict = self.__dict__.copy()
        
        # Convert quantum config to dict
        if isinstance(config_dict['quantum_config'], QuantumConfig):
            config_dict['quantum_config'] = config_dict['quantum_config'].__dict__
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def update(self, **kwargs) -> 'CSTConfig':
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self
    
    def get_quantum_state_space_size(self) -> int:
        """Calculate quantum state space dimensionality"""
        return 2 ** self.quantum_config.n_qubits
    
    def get_total_quantum_parameters(self) -> int:
        """Calculate total number of quantum parameters"""
        # Each layer has n_qubits * 3 (RX, RY, RZ) parameters
        return self.quantum_config.n_qubits * 3 * self.quantum_config.n_layers
    
    def estimate_quantum_advantage(self) -> Dict[str, Any]:
        """Estimate potential quantum advantage metrics"""
        quantum_state_dim = self.get_quantum_state_space_size()
        classical_equiv_params = quantum_state_dim ** 2
        quantum_params = self.get_total_quantum_parameters()
        
        return {
            'quantum_state_dimension': quantum_state_dim,
            'quantum_parameters': quantum_params,
            'classical_equivalent_parameters': classical_equiv_params,
            'parameter_efficiency': classical_equiv_params / quantum_params,
            'theoretical_speedup': f"{quantum_state_dim / self.quantum_config.n_qubits:.1f}x"
        }


@dataclass
class TrainingConfig:
    """Training-specific configuration with quantum support"""
    
    # Data paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 5000
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    wandb_project: Optional[str] = "cst-quantum"
    tensorboard_dir: str = "tensorboard_logs"
    
    # Distributed Training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    
    # Mixed Precision
    use_amp: bool = True
    amp_opt_level: str = "O1"
    
    # Quantum-Specific Training
    quantum_warmup_steps: int = 5000
    quantum_scheduler: str = "cosine"  # 'cosine', 'linear', 'constant'
    use_quantum_gradient_clipping: bool = True
    quantum_gradient_clip_norm: float = 0.5
    
    # Quantum Circuit Optimization
    optimize_quantum_circuits: bool = True
    circuit_optimization_level: int = 2  # 0-3, higher = more optimization
    compile_circuits_once: bool = True  # Compile circuits before training
    
    # Performance Tracking
    track_quantum_performance: bool = True
    log_quantum_circuit_metrics: bool = True
    save_quantum_states: bool = False  # Warning: memory intensive


@dataclass
class ProductionConfig:
    """Production deployment configuration with quantum support"""
    
    # Serving
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    health_check_interval: int = 60
    
    # Caching
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8080
    enable_tracing: bool = True
    trace_sampling_rate: float = 0.1
    
    # Resource Limits
    max_memory_usage: float = 0.8
    max_cpu_usage: float = 0.8
    
    # Model Serving
    model_path: str = "models/cst_model.pt"
    device: str = "cuda"
    precision: str = "fp16"
    
    # Quantum Production Settings
    use_quantum_in_production: bool = True
    quantum_fallback_classical: bool = True  # Fallback to classical if quantum fails
    quantum_timeout_ms: int = 100  # Timeout for quantum operations
    quantum_cache_results: bool = True  # Cache quantum circuit outputs
    quantum_batch_processing: bool = True  # Batch quantum operations
    
    # Quantum Backend for Production
    quantum_backend_type: str = "simulator"  # 'simulator', 'real_device', 'cloud'
    quantum_device_name: Optional[str] = None  # Specific device if using real hardware
    quantum_api_token: Optional[str] = None  # For cloud quantum services


# Default configurations
DEFAULT_QUANTUM_CONFIG = QuantumConfig()
DEFAULT_CST_CONFIG = CSTConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_PRODUCTION_CONFIG = ProductionConfig()


# Preset configurations for different use cases
class ConfigPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def get_research_config() -> CSTConfig:
        """Configuration optimized for research experiments"""
        config = CSTConfig()
        config.quantum_config.n_qubits = 8
        config.quantum_config.n_layers = 3
        config.quantum_config.shots = None  # Exact simulation
        config.quantum_config.track_quantum_metrics = True
        config.quantum_config.log_circuit_depth = True
        return config
    
    @staticmethod
    def get_production_config() -> CSTConfig:
        """Configuration optimized for production deployment"""
        config = CSTConfig()
        config.quantum_config.n_qubits = 6  # Smaller for speed
        config.quantum_config.n_layers = 2
        config.quantum_config.shots = 1000  # Faster sampling
        config.quantum_config.quantum_weight = 0.3  # Less quantum, more speed
        config.quantum_config.track_quantum_metrics = False
        return config
    
    @staticmethod
    def get_high_accuracy_config() -> CSTConfig:
        """Configuration optimized for maximum accuracy"""
        config = CSTConfig()
        config.quantum_config.n_qubits = 10
        config.quantum_config.n_layers = 4
        config.quantum_config.entanglement_pattern = 'full'
        config.quantum_config.use_data_reuploading = True
        config.quantum_config.quantum_weight = 0.7  # Favor quantum
        return config
    
    @staticmethod
    def get_fast_training_config() -> CSTConfig:
        """Configuration optimized for training speed"""
        config = CSTConfig()
        config.quantum_config.n_qubits = 6
        config.quantum_config.n_layers = 2
        config.quantum_config.entanglement_pattern = 'linear'  # Simpler
        config.quantum_config.use_data_reuploading = False
        config.batch_size = 64  # Larger batches
        return config


# Example usage
if __name__ == "__main__":
    # Create default config
    config = CSTConfig()
    
    # Print quantum advantage estimation
    advantage = config.estimate_quantum_advantage()
    print("Quantum Advantage Estimation:")
    for key, value in advantage.items():
        print(f"  {key}: {value}")
    
    # Save configuration
    config.to_yaml("configs/quantum_cst_config.yaml")
    
    # Load configuration
    loaded_config = CSTConfig.from_yaml("configs/quantum_cst_config.yaml")
    
    # Use presets
    research_config = ConfigPresets.get_research_config()
    production_config = ConfigPresets.get_production_config()
    
    print("\nConfigurations created successfully!")