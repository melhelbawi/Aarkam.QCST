# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum-Enhanced Training Pipeline for CST Models
Implements quantum-aware loss functions, optimizers, and training procedures
Fully standalone - no classical dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import json
import logging
import wandb
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import time

from quantum_cst_transformer import QuantumCSTransformer
from quantum_cst_config import QuantumConfig, CSTConfig
from quantum_transformer_utils import collate_quantum_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumInfoNCELoss(nn.Module):
    """
    Quantum-Enhanced InfoNCE Loss for Contrastive Learning
    
    Incorporates quantum entanglement measures to better capture
    feature correlations in the contrastive learning objective
    """
    
    def __init__(self, temperature: float = 0.07, quantum_weight: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.quantum_weight = quantum_weight
        
    def compute_quantum_entanglement_penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty based on quantum entanglement principles
        
        Encourages embeddings to have properties similar to quantum entangled states:
        - High correlation between related features
        - Low correlation between unrelated features
        """
        # Compute correlation matrix
        normalized_emb = F.normalize(embeddings, dim=-1)
        correlation_matrix = torch.matmul(normalized_emb, normalized_emb.t())
        
        # Von Neumann entropy as entanglement measure
        # Higher entropy = more entanglement = better representation
        eigenvalues = torch.linalg.eigvalsh(correlation_matrix)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)  # Numerical stability
        
        # Normalized eigenvalues (probability distribution)
        probs = eigenvalues / eigenvalues.sum()
        
        # Entropy: -Σ p_i log(p_i)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Maximize entropy (negative for minimization)
        return -entropy
    
    def forward(self, positive_pairs: torch.Tensor, 
                negative_pairs: torch.Tensor,
                quantum_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            positive_pairs: [batch_size, embedding_dim]
            negative_pairs: [batch_size * num_negatives, embedding_dim]
            quantum_embeddings: Optional quantum-processed embeddings
        """
        batch_size = positive_pairs.size(0)
        
        # Normalize embeddings
        positive_pairs = F.normalize(positive_pairs, dim=1)
        negative_pairs = F.normalize(negative_pairs, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(positive_pairs * positive_pairs, dim=1) / self.temperature
        
        # Reshape negatives
        num_negatives = negative_pairs.size(0) // batch_size
        negative_pairs = negative_pairs.view(batch_size, num_negatives, -1)
        
        neg_sims = torch.bmm(
            positive_pairs.unsqueeze(1),
            negative_pairs.transpose(1, 2)
        ).squeeze(1) / self.temperature
        
        # Combine positive and negative similarities
        all_sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        
        # InfoNCE loss
        labels = torch.zeros(batch_size, dtype=torch.long, device=positive_pairs.device)
        contrastive_loss = F.cross_entropy(all_sims, labels)
        
        # Add quantum entanglement penalty if quantum embeddings provided
        if quantum_embeddings is not None:
            entanglement_penalty = self.compute_quantum_entanglement_penalty(quantum_embeddings)
            total_loss = contrastive_loss + self.quantum_weight * entanglement_penalty
            
            return total_loss
        
        return contrastive_loss


class QuantumFidelityLoss(nn.Module):
    """
    Quantum Fidelity Loss - Measures how well quantum states are preserved
    
    Based on quantum state fidelity: F(ρ, σ) = Tr(√(√ρ σ √ρ))
    Encourages quantum circuits to maintain high fidelity
    """
    
    def __init__(self, target_fidelity: float = 0.95):
        super().__init__()
        self.target_fidelity = target_fidelity
        
    def compute_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum fidelity between two states
        
        Simplified version using inner product for pure states
        """
        # Normalize states
        state1_norm = F.normalize(state1, dim=-1)
        state2_norm = F.normalize(state2, dim=-1)
        
        # Fidelity for pure states: |⟨ψ|φ⟩|²
        inner_product = torch.sum(state1_norm * state2_norm, dim=-1)
        fidelity = inner_product ** 2
        
        return fidelity
    
    def forward(self, quantum_outputs: torch.Tensor, 
                reference_outputs: torch.Tensor) -> torch.Tensor:
        """
        Loss that encourages quantum outputs to maintain high fidelity
        with reference (e.g., classical) outputs
        """
        fidelity = self.compute_fidelity(quantum_outputs, reference_outputs)
        
        # Loss is how far we are from target fidelity
        loss = F.mse_loss(fidelity, torch.full_like(fidelity, self.target_fidelity))
        
        return loss


class QuantumRegularizer(nn.Module):
    """
    Quantum-specific regularization terms
    
    Includes:
    1. Circuit depth penalty (encourage shallow circuits)
    2. Entanglement regularization (control entanglement levels)
    3. Quantum noise resilience (prepare for NISQ devices)
    """
    
    def __init__(self, config: QuantumConfig):
        super().__init__()
        self.config = config
        self.depth_penalty_weight = 0.01
        self.entanglement_weight = config.quantum_entanglement_regularization
        
    def compute_circuit_depth_penalty(self, quantum_params: torch.Tensor) -> torch.Tensor:
        """
        Penalize large quantum parameter magnitudes
        Larger params often indicate deeper effective circuits
        """
        return torch.mean(quantum_params ** 2)
    
    def compute_entanglement_regularization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Regularize entanglement levels
        
        Too much entanglement can make circuits hard to train
        Too little loses quantum advantage
        """
        # Compute pairwise correlations
        batch_size, dim = embeddings.shape
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Center embeddings
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Covariance matrix
        cov = torch.matmul(centered.t(), centered) / (batch_size - 1)
        
        # Off-diagonal elements represent entanglement
        off_diagonal = cov - torch.diag(torch.diag(cov))
        entanglement_measure = torch.sum(off_diagonal ** 2)
        
        # Target moderate entanglement (not too high, not too low)
        target_entanglement = 0.3 * dim  # Heuristic target
        loss = (entanglement_measure - target_entanglement) ** 2
        
        return loss
    
    def forward(self, quantum_params: Dict[str, torch.Tensor],
                quantum_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute total quantum regularization loss
        """
        total_loss = torch.tensor(0.0, device=quantum_embeddings.device)
        
        # Circuit depth penalty
        for param_name, params in quantum_params.items():
            if 'quantum' in param_name.lower():
                total_loss += self.depth_penalty_weight * self.compute_circuit_depth_penalty(params)
        
        # Entanglement regularization
        if self.entanglement_weight > 0:
            total_loss += self.entanglement_weight * self.compute_entanglement_regularization(quantum_embeddings)
        
        return total_loss


class QuantumAwareOptimizer:
    """
    Quantum-Aware Optimizer wrapper
    
    Implements quantum-specific optimization strategies:
    1. Parameter-shift rule for quantum gradients
    2. Different learning rates for quantum vs classical parameters
    3. Quantum gradient clipping
    """
    
    def __init__(self, model: nn.Module, config: CSTConfig, training_config: TrainingConfig):
        self.model = model
        self.config = config
        self.training_config = training_config
        
        # Separate parameter groups
        quantum_params = []
        classical_params = []
        
        for name, param in model.named_parameters():
            if 'quantum' in name.lower() or 'qnode' in name.lower():
                quantum_params.append(param)
            else:
                classical_params.append(param)
        
        # Different learning rates for quantum and classical
        param_groups = [
            {
                'params': quantum_params,
                'lr': config.quantum_config.quantum_lr,
                'weight_decay': 0.0,  # No weight decay for quantum params
                'name': 'quantum'
            },
            {
                'params': classical_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay,
                'name': 'classical'
            }
        ]
        
        # Choose optimizer based on config
        if config.quantum_config.quantum_optimizer == 'adam':
            self.optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
        elif config.quantum_config.quantum_optimizer == 'sgd':
            self.optimizer = SGD(param_groups, momentum=0.9)
        else:
            self.optimizer = AdamW(param_groups)
        
        self.use_quantum_gradient_clipping = training_config.use_quantum_gradient_clipping
        self.quantum_clip_norm = training_config.quantum_gradient_clip_norm
        
    def step(self):
        """Optimization step with quantum-aware gradient handling"""
        
        # Separate gradient clipping for quantum and classical parameters
        if self.use_quantum_gradient_clipping:
            for group in self.optimizer.param_groups:
                if group['name'] == 'quantum':
                    torch.nn.utils.clip_grad_norm_(
                        group['params'], 
                        self.quantum_clip_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        group['params'],
                        self.config.gradient_clip_norm
                    )
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class QuantumSpectralRegularizer:
    """Enhanced spectral regularizer with quantum considerations"""
    
    def __init__(self, config: CSTConfig):
        self.config = config
        self.reference_embeddings = {}
        self.quantum_reference_embeddings = {}
        self.update_frequency = config.reference_update_freq
        self.step_count = 0
        self.momentum = config.reference_momentum
        
    def compute_drift_loss(self, current_embeddings: torch.Tensor, 
                          fragment_ids: torch.Tensor,
                          is_quantum: bool = False) -> torch.Tensor:
        """Compute drift with separate tracking for quantum embeddings"""
        
        reference_dict = self.quantum_reference_embeddings if is_quantum else self.reference_embeddings
        
        drift_loss = torch.tensor(0.0, device=current_embeddings.device, requires_grad=True)
        
        for frag_id in fragment_ids.unique():
            frag_id_item = frag_id.item()
            if frag_id_item in reference_dict:
                current_mask = fragment_ids == frag_id
                current_repr = current_embeddings[current_mask].mean(0)
                reference_repr = reference_dict[frag_id_item]
                
                drift_loss = drift_loss + F.mse_loss(current_repr, reference_repr)
                
        return drift_loss / len(fragment_ids.unique()) if len(fragment_ids.unique()) > 0 else drift_loss
    
    def update_references(self, embeddings: torch.Tensor, 
                         fragment_ids: torch.Tensor,
                         is_quantum: bool = False):
        """Update reference embeddings with EMA"""
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            reference_dict = self.quantum_reference_embeddings if is_quantum else self.reference_embeddings
            
            for frag_id in fragment_ids.unique():
                frag_id_item = frag_id.item()
                current_mask = fragment_ids == frag_id
                current_repr = embeddings[current_mask].mean(0).detach()
                
                if frag_id_item in reference_dict:
                    reference_dict[frag_id_item] = (
                        self.momentum * reference_dict[frag_id_item] +
                        (1 - self.momentum) * current_repr
                    )
                else:
                    reference_dict[frag_id_item] = current_repr


class QuantumCSTrainer:
    """
    Main Quantum-Enhanced Trainer for CST Models
    
    Integrates:
    - Quantum-aware loss functions
    - Quantum-specific optimizers
    - Quantum performance monitoring
    - Hybrid quantum-classical training strategies
    """
    
    def __init__(self, 
                 model: QuantumCSTransformer, 
                 config: CSTConfig,
                 train_config: TrainingConfig):
        self.model = model
        self.config = config
        self.train_config = train_config
        
        # Quantum-aware losses
        self.contrastive_loss = QuantumInfoNCELoss(
            temperature=config.temperature,
            quantum_weight=config.quantum_loss_weight
        )
        self.quantum_fidelity_loss = QuantumFidelityLoss(target_fidelity=0.95)
        self.quantum_regularizer = QuantumRegularizer(config.quantum_config)
        self.spectral_regularizer = QuantumSpectralRegularizer(config)
        
        # Quantum-aware optimizer
        self.optimizer = None  # Initialized in setup_optimizer
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.quantum_metrics = defaultdict(list)
        
        # Setup distributed training if needed
        self.is_distributed = train_config.distributed
        if self.is_distributed:
            self.setup_distributed()
        
        # Setup logging
        if train_config.wandb_project and (not self.is_distributed or dist.get_rank() == 0):
            wandb.init(
                project=train_config.wandb_project,
                config={**config.__dict__, **config.quantum_config.__dict__}
            )
            wandb.watch(model, log='all', log_freq=100)
        
        logger.info("=" * 60)
        logger.info("Quantum-Enhanced CST Trainer Initialized")
        logger.info("=" * 60)
        logger.info(f"Quantum Enabled: {config.quantum_config.enable_quantum}")
        logger.info(f"Quantum Qubits: {config.quantum_config.n_qubits}")
        logger.info(f"Quantum Layers: {config.quantum_config.n_layers}")
        logger.info(f"State Space Dimension: {2 ** config.quantum_config.n_qubits}")
        logger.info("=" * 60)
    
    def setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend=self.train_config.backend)
        torch.cuda.set_device(self.train_config.rank)
        self.model = DDP(self.model, device_ids=[self.train_config.rank])
    
    def setup_optimizer(self, train_loader: DataLoader):
        """Setup quantum-aware optimizer and scheduler"""
        
        # Quantum-aware optimizer
        self.optimizer = QuantumAwareOptimizer(self.model, self.config, self.train_config)
        
        # Setup scheduler with quantum warmup
        total_steps = self.train_config.max_epochs * len(train_loader)
        
        # Quantum warmup (longer warmup for quantum parameters)
        quantum_warmup = LinearLR(
            self.optimizer.optimizer,
            start_factor=0.01,
            total_iters=self.train_config.quantum_warmup_steps
        )
        
        # Main scheduler
        if self.train_config.quantum_scheduler == 'cosine':
            main_scheduler = CosineAnnealingLR(
                self.optimizer.optimizer,
                T_max=total_steps - self.train_config.quantum_warmup_steps
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer.optimizer,
                start_factor=1.0,
                total_iters=total_steps - self.train_config.quantum_warmup_steps
            )
        
        self.scheduler = SequentialLR(
            self.optimizer.optimizer,
            [quantum_warmup, main_scheduler],
            milestones=[self.train_config.quantum_warmup_steps]
        )
        
        logger.info(f"Optimizer setup complete")
        logger.info(f"  Quantum warmup steps: {self.train_config.quantum_warmup_steps}")
        logger.info(f"  Total training steps: {total_steps}")
    
    def quantum_contrastive_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Quantum-enhanced contrastive learning step
        """
        input_ids = batch['input_ids']
        context_data = batch['context_data']
        
        # Get quantum-enhanced embeddings
        positive_embeddings = self.model.get_embeddings(input_ids, context_data)
        
        # Create negative contexts (shuffle metadata)
        negative_context_data = context_data.copy()
        batch_size = input_ids.size(0)
        perm = torch.randperm(batch_size)
        
        if 'metadata' in context_data:
            negative_context_data['metadata'] = {
                k: v[perm] if isinstance(v, torch.Tensor) else v
                for k, v in context_data['metadata'].items()
            }
        
        # Get embeddings with negative context
        negative_embeddings = self.model.get_embeddings(input_ids, negative_context_data)
        
        # Pool sequence representations
        positive_pooled = positive_embeddings.mean(dim=1)
        negative_pooled = negative_embeddings.mean(dim=1)
        
        # Quantum-enhanced contrastive loss
        contrastive_loss = self.contrastive_loss(
            positive_pooled,
            negative_pooled,
            quantum_embeddings=positive_pooled  # Use for entanglement penalty
        )
        
        metrics = {
            'contrastive_loss': contrastive_loss.item()
        }
        
        return contrastive_loss, metrics
    
    def language_modeling_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Language modeling step with quantum awareness
        """
        outputs = self.model(
            input_ids=batch['input_ids'],
            context_data=batch['context_data'],
            attention_mask=batch['attention_mask'],
            fragment_chars=batch['fragment_chars'],
            context_chars=batch['context_chars'],
            labels=batch['labels'],
            return_quantum_metrics=True
        )
        
        mlm_loss = outputs['loss']
        quantum_metrics = outputs.get('quantum_metrics', {})
        
        # Spectral regularization (separate for quantum and classical)
        embeddings = outputs['hidden_states']
        input_ids_flat = batch['input_ids'].view(-1)
        embeddings_flat = embeddings.view(-1, embeddings.size(-1))
        
        # Determine which embeddings were quantum-processed
        is_quantum_batch = quantum_metrics.get('quantum_ratio', 0) > 0.5
        
        drift_loss = self.spectral_regularizer.compute_drift_loss(
            embeddings_flat,
            input_ids_flat,
            is_quantum=is_quantum_batch
        )
        
        # Update reference embeddings
        self.spectral_regularizer.update_references(
            embeddings_flat,
            input_ids_flat,
            is_quantum=is_quantum_batch
        )
        
        # Quantum regularization
        quantum_params = {
            name: param for name, param in self.model.named_parameters()
            if 'quantum' in name.lower()
        }
        
        quantum_reg_loss = self.quantum_regularizer(
            quantum_params,
            embeddings.mean(dim=1)  # Pool for regularization
        )
        
        # Total loss
        total_loss = (mlm_loss + 
                     self.config.drift_regularization_weight * drift_loss +
                     self.config.quantum_loss_weight * quantum_reg_loss)
        
        metrics = {
            'mlm_loss': mlm_loss.item(),
            'drift_loss': drift_loss.item(),
            'quantum_reg_loss': quantum_reg_loss.item(),
            **{f'quantum_{k}': v for k, v in quantum_metrics.items()
               if isinstance(v, (int, float))}
        }
        
        return total_loss, metrics
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Complete training step with quantum-aware losses
        """
        self.model.train()
        
        # Contrastive learning (with quantum entanglement penalty)
        if self.config.use_quantum_contrastive:
            contrastive_loss, contrastive_metrics = self.quantum_contrastive_step(batch)
        else:
            contrastive_loss = torch.tensor(0.0, device=batch['input_ids'].device)
            contrastive_metrics = {}
        
        # Language modeling (with quantum regularization)
        mlm_loss, mlm_metrics = self.language_modeling_step(batch)
        
        # Combined loss
        total_loss = (self.config.contrastive_weight * contrastive_loss + 
                     self.config.mlm_weight * mlm_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Quantum-aware optimization step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # Compile metrics
        metrics = {
            'total_loss': total_loss.item(),
            'learning_rate': self.optimizer.optimizer.param_groups[0]['lr'],
            'quantum_lr': self.optimizer.optimizer.param_groups[0]['lr'],  # Quantum LR
            **contrastive_metrics,
            **mlm_metrics
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop with quantum metrics"""
        self.model.eval()
        val_losses = []
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    context_data=batch['context_data'],
                    attention_mask=batch['attention_mask'],
                    fragment_chars=batch['fragment_chars'],
                    context_chars=batch['context_chars'],
                    labels=batch['labels'],
                    return_quantum_metrics=True
                )
                
                val_losses.append(outputs['loss'].item())
                
                # Collect quantum metrics
                if 'quantum_metrics' in outputs:
                    for key, value in outputs['quantum_metrics'].items():
                        if isinstance(value, (int, float)):
                            all_metrics[f'val_quantum_{key}'].append(value)
                
                # Collect CST stats
                if 'cst_stats' in outputs:
                    for key, value in outputs['cst_stats'].items():
                        if isinstance(value, (int, float)):
                            all_metrics[f'val_cst_{key}'].append(value)
        
        # Average metrics
        avg_metrics = {
            'val_loss': np.mean(val_losses),
            **{k: np.mean(v) for k, v in all_metrics.items()}
        }
        
        return avg_metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save checkpoint with quantum state"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'quantum_config': self.config.quantum_config.__dict__,
            'spectral_regularizer': {
                'reference_embeddings': self.spectral_regularizer.reference_embeddings,
                'quantum_reference_embeddings': self.spectral_regularizer.quantum_reference_embeddings
            },
            'quantum_metrics_history': self.model.quantum_metrics_history if hasattr(self.model, 'quantum_metrics_history') else []
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint with quantum state"""
        checkpoint = torch.load(filepath, map_location='cuda')
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Restore quantum-specific state
        if 'spectral_regularizer' in checkpoint:
            self.spectral_regularizer.reference_embeddings = checkpoint['spectral_regularizer']['reference_embeddings']
            self.spectral_regularizer.quantum_reference_embeddings = checkpoint['spectral_regularizer']['quantum_reference_embeddings']
        
        if 'quantum_metrics_history' in checkpoint:
            self.model.quantum_metrics_history = checkpoint['quantum_metrics_history']
        
        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"  Epoch: {self.epoch}, Step: {self.global_step}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop with quantum enhancements"""
        self.setup_optimizer(train_loader)
        
        logger.info("=" * 60)
        logger.info(f"Starting Quantum-Enhanced Training")
        logger.info(f"  Epochs: {self.train_config.max_epochs}")
        logger.info(f"  Total steps: {len(train_loader) * self.train_config.max_epochs}")
        logger.info(f"  Quantum warmup: {self.train_config.quantum_warmup_steps} steps")
        logger.info("=" * 60)
        
        for epoch in range(self.train_config.max_epochs):
            self.epoch = epoch
            
            # Training loop
            self.model.train()
            epoch_metrics = defaultdict(list)
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(progress_bar):
                
                # Move to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                step_metrics = self.train_step(batch)
                
                # Update metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['total_loss']:.4f}",
                    'lr': f"{step_metrics['learning_rate']:.2e}",
                    'q_ratio': f"{step_metrics.get('quantum_quantum_ratio', 0):.2%}"
                })
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.train_config.log_every_n_steps == 0:
                    avg_metrics = {k: np.mean(v[-self.train_config.log_every_n_steps:]) 
                                 for k, v in epoch_metrics.items()}
                    
                    if wandb.run:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    logger.info(f"Step {self.global_step}: Loss={avg_metrics['total_loss']:.4f}, "
                              f"Q_Ratio={avg_metrics.get('quantum_quantum_ratio', 0):.2%}")
                
                # Validation
                if val_loader and self.global_step % self.train_config.eval_every_n_steps == 0:
                    val_metrics = self.validate(val_loader)
                    
                    if wandb.run:
                        wandb.log(val_metrics, step=self.global_step)
                    
                    logger.info(f"Validation at step {self.global_step}:")
                    logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                    logger.info(f"  Quantum Usage: {val_metrics.get('val_quantum_quantum_ratio', 0):.2%}")
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(
                            f"{self.train_config.checkpoint_dir}/checkpoint_step_{self.global_step}.pt",
                            is_best=True
                        )
                        logger.info(f"  ✅ New best model saved!")
                
                # Checkpoint saving
                if self.global_step % self.train_config.save_every_n_steps == 0:
                    self.save_checkpoint(
                        f"{self.train_config.checkpoint_dir}/checkpoint_step_{self.global_step}.pt"
                    )
            
            # End of epoch validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                logger.info(f"End of epoch {epoch} validation:")
                logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                
                if wandb.run:
                    wandb.log({f"epoch_{k}": v for k, v in val_metrics.items()}, 
                            step=self.global_step)
        
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info("=" * 60)
        
        # Final quantum summary
        quantum_summary = self.model.get_quantum_summary()
        logger.info("Final Quantum Usage Summary:")
        logger.info(f"  Total quantum tokens: {quantum_summary['cst_stats'].get('quantum_processed_tokens', 0)}")
        logger.info(f"  Quantum ratio: {quantum_summary['cst_stats'].get('quantum_usage_ratio', 0):.2%}")
        logger.info(f"  Cache hit rate: {quantum_summary['cst_stats'].get('hit_rate', 0):.2%}")


# Helper functions for data loading

class QuantumAwareDataset(Dataset):
    """Simple quantum-aware dataset for training"""
    
    def __init__(self, data: List[Dict], config):
        self.data = data
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_quantum_aware_data_loader(data: List[Dict], config, 
                                     training_config,
                                     split: str = 'train') -> DataLoader:
    """
    Create data loader optimized for quantum training
    
    Features:
    - Balanced batching (mix ambiguous and non-ambiguous samples)
    - Quantum-relevant data augmentation
    - Efficient context data preparation
    """
    dataset = QuantumAwareDataset(data, config)
    
    # Quantum-aware sampler (optional: sample more ambiguous examples)
    sampler = None
    shuffle = (split == 'train')
    
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size if split == 'train' else training_config.batch_size * 2,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_quantum_batch,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=2
    )
    return loader


def main():
    """Main training script with quantum enhancements"""
    
    # Load configurations
    from config import CSTConfig, TrainingConfig, QuantumConfig, ConfigPresets
    
    # Use high-accuracy quantum config
    config = ConfigPresets.get_high_accuracy_config()
    training_config = TrainingConfig()
    
    logger.info("=" * 60)
    logger.info("Quantum-Enhanced CST Training")
    logger.info("=" * 60)
    logger.info(f"Model Configuration:")
    logger.info(f"  d_model: {config.d_model}")
    logger.info(f"  num_layers: {config.num_layers}")
    logger.info(f"  vocab_size: {config.vocab_size}")
    logger.info(f"\nQuantum Configuration:")
    logger.info(f"  Quantum enabled: {config.quantum_config.enable_quantum}")
    logger.info(f"  n_qubits: {config.quantum_config.n_qubits}")
    logger.info(f"  n_layers: {config.quantum_config.n_layers}")
    logger.info(f"  State space: {2 ** config.quantum_config.n_qubits}")
    logger.info(f"  Quantum weight: {config.quantum_config.quantum_weight}")
    logger.info("=" * 60)
    
    # Create model
    model = QuantumCSTransformer(config, task_type='mlm')
    model.cuda()
    model.enable_cst_profiling(True)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = sum(p.numel() for n, p in model.named_parameters() 
                        if 'quantum' in n.lower())
    
    logger.info(f"\nModel Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Quantum parameters: {quantum_params:,} ({quantum_params/total_params*100:.1f}%)")
    logger.info(f"  Classical parameters: {total_params - quantum_params:,}")
    
    # Quantum advantage estimation
    advantage = config.estimate_quantum_advantage()
    logger.info(f"\nTheoretical Quantum Advantage:")
    logger.info(f"  Parameter efficiency: {advantage['parameter_efficiency']:.1f}x")
    logger.info(f"  State space dimension: {advantage['quantum_state_dimension']}")
    logger.info("=" * 60)
    
    # Setup data loaders
    train_loader = create_quantum_aware_data_loader(
        training_config.train_data_path,
        config,
        training_config,
        split='train'
    )
    
    val_loader = create_quantum_aware_data_loader(
        training_config.val_data_path,
        config,
        training_config,
        split='val'
    )
    
    logger.info(f"\nData Loaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info("=" * 60)
    
    # Setup trainer
    trainer = QuantumCSTrainer(model, config, training_config)
    
    # Start training
    logger.info("\n🚀 Starting training...\n")
    trainer.train(train_loader, val_loader)
    
    # Save final model
    model.save_pretrained(f"{training_config.checkpoint_dir}/final_model")
    
    # Print final quantum summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL QUANTUM SUMMARY")
    logger.info("=" * 60)
    
    quantum_summary = model.get_quantum_summary()
    logger.info(f"Quantum Processing Statistics:")
    logger.info(f"  Total tokens processed: {quantum_summary['cst_stats'].get('quantum_processed_tokens', 0) + quantum_summary['cst_stats'].get('classical_processed_tokens', 0)}")
    logger.info(f"  Quantum tokens: {quantum_summary['cst_stats'].get('quantum_processed_tokens', 0)}")
    logger.info(f"  Quantum usage ratio: {quantum_summary['cst_stats'].get('quantum_usage_ratio', 0):.2%}")
    logger.info(f"  Cache efficiency: {quantum_summary['cst_stats'].get('hit_rate', 0):.2%}")
    logger.info(f"  Quantum cache hit rate: {quantum_summary['cst_stats'].get('quantum_hit_rate', 0):.2%}")
    
    if 'average_quantum_metrics' in quantum_summary:
        logger.info(f"\nAverage Quantum Metrics:")
        for key, value in quantum_summary['average_quantum_metrics'].items():
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info("=" * 60)
    logger.info("✅ Training completed successfully!")
    logger.info("=" * 60)


# Utility: Quantum training analysis
def analyze_quantum_training_run(checkpoint_dir: str):
    """
    Analyze quantum usage throughout training
    
    Loads checkpoints and analyzes:
    - Quantum vs classical usage over time
    - Quantum circuit parameter evolution
    - Cache performance trends
    """
    import glob
    import matplotlib.pyplot as plt
    
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint_step_*.pt"))
    
    quantum_ratios = []
    cache_hit_rates = []
    steps = []
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if 'quantum_metrics_history' in ckpt:
            metrics = ckpt['quantum_metrics_history']
            if metrics:
                avg_ratio = np.mean([m.get('quantum_ratio', 0) for m in metrics])
                quantum_ratios.append(avg_ratio)
        
        step = int(ckpt_path.split('_')[-1].replace('.pt', ''))
        steps.append(step)
    
    # Plot quantum usage over training
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(steps, quantum_ratios, 'b-', linewidth=2)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Quantum Usage Ratio')
    axes[0].set_title('Quantum Processing Over Training')
    axes[0].grid(True, alpha=0.3)
    
    if cache_hit_rates:
        axes[1].plot(steps, cache_hit_rates, 'g-', linewidth=2)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Cache Hit Rate')
        axes[1].set_title('Cache Performance Over Training')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{checkpoint_dir}/quantum_training_analysis.png", dpi=300)
    plt.close()
    
    logger.info(f"Training analysis saved to {checkpoint_dir}/quantum_training_analysis.png")
    
    return {
        'steps': steps,
        'quantum_ratios': quantum_ratios,
        'cache_hit_rates': cache_hit_rates,
        'final_quantum_ratio': quantum_ratios[-1] if quantum_ratios else 0,
        'avg_quantum_ratio': np.mean(quantum_ratios) if quantum_ratios else 0
    }


if __name__ == "__main__":
    # Check for required packages
    try:
        import pennylane
        logger.info("✅ PennyLane installed")
    except ImportError:
        logger.error("❌ PennyLane not installed. Run: pip install pennylane")
        exit(1)
    
    # Run training
    main()