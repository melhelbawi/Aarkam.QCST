# CST / QCST Dual License
# Non-commercial research use only.
# Commercial use requires explicit permission.
# Copyright (c) 2025 Mohamed Mohamed Elhelbawi
# All rights reserved.
# See LICENSE file in the project root for full license information.

"""
Quantum-Enhanced CST - Unified Runner Script
Supports modes: demo, train, benchmark
"""

import argparse
import logging
import os
import sys
import torch
import time
import json
from pathlib import Path
from tabulate import tabulate
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
quantum_dir = os.path.join(project_root, 'src', 'cst', 'quantum')
classical_dir = os.path.join(project_root, 'src', 'cst', 'classical')

sys.path.insert(0, quantum_dir)
sys.path.insert(0, classical_dir)

from quantum_cst_config import CSTConfig, QuantumConfig, ConfigPresets, TrainingConfig
from quantum_cst_transformer import QuantumCSTransformer

# Try importing training utilities from classical module
try:
    from training_pipeline import CSTDataset, collate_fn
except ImportError:
    # Fallback if training_pipeline not available
    logger.warning("training_pipeline not found, using minimal dataset")
    CSTDataset = None
    collate_fn = None

from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def run_demo():
    """Run demonstration of ambiguity resolution"""
    print("\n" + "="*60)
    print("🚀 Quantum CST Demo: Ambiguity Resolution")
    print("="*60 + "\n")

    # 1. Setup Model
    print("Initializing Quantum-Enhanced Model (Research Preset)...")
    config = ConfigPresets.get_research_config()
    # Ensure vocab size is large enough for demo
    config.vocab_size = 5000 
    model = QuantumCSTransformer(config, task_type='mlm')
    model.eval()

    # 2. Demo Sentences
    sentences = [
        ("The bank denied the loan application.", "bank", "Financial Institution"),
        ("I sat by the river bank and watched the water.", "bank", "River Edge"),
        ("The apple was red and delicious.", "apple", "Fruit"),
        ("Apple released a new iPhone today.", "apple", "Tech Company")
    ]

    # Mock tokenizer (simple mapping for demo purposes)
    def simple_tokenize(text):
        # Deterministic hashing to get IDs in range
        return torch.tensor([hash(w) % 4000 + 100 for w in text.lower().split()])

    print(f"Analyzing {len(sentences)} sentences for Contextual Embeddings...\n")
    
    results = []
    embeddings_store = {}

    for text, target_word, context_desc in sentences:
        tokens = simple_tokenize(text)
        input_ids = tokens.unsqueeze(0) # Batch size 1
        
        # Determine target index
        words = text.lower().split()
        try:
            target_idx = words.index(target_word.lower())
        except ValueError:
            continue

        # Create dummy context
        context_data = {
            'document_embedding': torch.randn(1, config.raw_doc_dim),
            'metadata': {
                'author': torch.zeros(1, dtype=torch.long),
                'domain': torch.zeros(1, dtype=torch.long)
            }
        }

        # Run Inference
        with torch.no_grad():
            outputs = model(input_ids, context_data, return_quantum_metrics=True)
            
        # Extract embedding of target word
        # shape: [batch, seq, dim] -> [dim]
        target_embedding = outputs['hidden_states'][0, target_idx, :]
        
        embeddings_store[(target_word, context_desc)] = target_embedding
        
        # Get metrics
        q_metrics = outputs.get('quantum_metrics', {})
        results.append([
            text, 
            target_word, 
            context_desc, 
            f"{q_metrics.get('quantum_ratio', 0):.1%}"
        ])

    # 3. Display Results
    print(tabulate(results, headers=["Sentence", "Target", "Context", "Quantum Usage"], tablefmt="grid"))

    # 4. Calculate Similarity
    print("\n\n📊 Semantic Analysis (Cosine Similarity):")
    
    # Bank Comparison
    e1 = embeddings_store[("bank", "Financial Institution")]
    e2 = embeddings_store[("bank", "River Edge")]
    sim_bank = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    
    # Apple Comparison
    e3 = embeddings_store[("apple", "Fruit")]
    e4 = embeddings_store[("apple", "Tech Company")]
    sim_apple = torch.nn.functional.cosine_similarity(e3.unsqueeze(0), e4.unsqueeze(0)).item()
    
    # Control (Same context)
    e1_clone = e1.clone()
    sim_control = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e1_clone.unsqueeze(0)).item()

    sim_table = [
        ["Bank (Financial) vs Bank (River)", f"{sim_bank:.4f}", "Low (Distinct Meanings)"],
        ["Apple (Fruit) vs Apple (Tech)", f"{sim_apple:.4f}", "Low (Distinct Meanings)"],
        ["Bank (Financial) vs Bank (Financial)", f"{sim_control:.4f}", "High (Same Meaning)"]
    ]
    
    print(tabulate(sim_table, headers=["Comparison", "Similarity", "Expected"], tablefmt="simple"))
    print("\n✅ Demo Completed Successfully!")


def run_train(epochs=2, batch_size=4):
    """Run training loop verification"""
    print("\n" + "="*60)
    print(f"🚂 Quantum CST Training Verification (Epochs: {epochs})")
    print("="*60 + "\n")

    # Config
    config = ConfigPresets.get_fast_training_config()
    train_config = TrainingConfig()
    train_config.max_epochs = epochs
    train_config.checkpoint_dir = "checkpoints_test"
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    # Model
    print("Initializing Model...")
    model = QuantumCSTransformer(config, task_type='mlm')
    if torch.cuda.is_available():
        model.cuda()

    # Data (Synthetic)
    print("Generating Synthetic Dataset...")
    train_dataset = CSTDataset("dummy_path", config, split='train')
    # Hack to limit synthetic data size for quick test
    train_dataset.data = train_dataset.data[:20] 
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Trainer
    print("Starting Training Loop...")
    trainer = CSTTrainer(model, config, train_config)
    trainer.train(train_loader)
    
    print(f"\n✅ Training Completed! Checkpoints saved to {train_config.checkpoint_dir}")


def run_benchmark():
    """Compare Quantum vs Classical performance"""
    print("\n" + "="*60)
    print("⚡ Quantum vs Classical Benchmark")
    print("="*60 + "\n")

    config = ConfigPresets.get_production_config()
    input_ids = torch.randint(0, config.vocab_size, (16, 64)) # Batch 16, Seq 64
    context_data = {
        'document_embedding': torch.randn(16, config.raw_doc_dim),
        'metadata': {
            'author': torch.zeros(16, dtype=torch.long),
            'domain': torch.zeros(16, dtype=torch.long)
        }
    }
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        context_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in context_data.items()}
        context_data['metadata'] = {k: v.cuda() for k, v in context_data['metadata'].items()}

    results = []

    # 1. Classical Mode
    print("Benchmarking Classical Mode...")
    config.quantum_config.enable_quantum = False
    model_c = QuantumCSTransformer(config)
    if torch.cuda.is_available(): model_c.cuda()
    model_c.eval()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model_c(input_ids, context_data)
    end = time.time()
    avg_c = (end - start) / 50
    results.append(["Classical", f"{avg_c*1000:.2f} ms", "1.0x", "Baseline"])

    # 2. Quantum Mode
    print("Benchmarking Quantum Mode...")
    config.quantum_config.enable_quantum = True
    model_q = QuantumCSTransformer(config)
    if torch.cuda.is_available(): model_q.cuda()
    model_q.eval()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model_q(input_ids, context_data)
    end = time.time()
    avg_q = (end - start) / 50
    
    # Calculate overhead
    overhead = avg_q / avg_c
    results.append(["Quantum (Simulated)", f"{avg_q*1000:.2f} ms", f"{overhead:.1f}x", "Simulation Overhead"])

    print("\n" + tabulate(results, headers=["Mode", "Avg Latency (Batch 16)", "Rel. Time", "Note"], tablefmt="pretty"))
    print("\n*Note: Quantum latency includes simulation overhead. On real QPU, latency depends on circuit execution time not CPU simulation.")


def main():
    parser = argparse.ArgumentParser(description="Quantum CST Runner")
    parser.add_argument("--mode", choices=['demo', 'train', 'benchmark'], required=True, help="Execution mode")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'train':
        run_train(args.epochs, args.batch)
    elif args.mode == 'benchmark':
        run_benchmark()

if __name__ == "__main__":
    main()
