# User Execution Plan - Quantum CST

This plan details the creation of a unified execution script `run_cst.py` to allow the user to easily Train, Evaluate, and Demonstrate the Quantum-Enhanced Contextual Spectrum Tokenization system.

## Goal
Provide a "One-Click" experience for the user to see the value of their project.

## Proposed Changes

### [NEW] `src/cst/quantum/run_cst.py`
This script will serve as the main entry point. It will support three modes via CLI arguments:

1.  **`--mode demo`**:
    *   Loads a pre-configured "Research" model.
    *   Runs inference on a set of hard-coded "ambiguous sentences" (e.g., "The bank denied the loan" vs "I sat by the river bank").
    *   **Visualizes** the difference in token embeddings (cosine similarity) and displays Quantum Metrics (qubit usage, circuit depth).
    *   **Goal**: Show "Smart Tokenization" in action.

2.  **`--mode train`**:
    *   Initializes a `QuantumCSTransformer` with a "Fast Training" preset.
    *   Uses the `CSTDataset` (synthetic data generator) from `training_pipeline.py`.
    *   Runs a short training loop (default 2 epochs) with a progress bar.
    *   Saves the model to `checkpoints/`.
    *   **Goal**: Prove the pipeline works end-to-end.

3.  **`--mode benchmark`**:
    *   Compare **Classical CST** vs **Quantum CST**.
    *   Runs the same forward pass on both models.
    *   Reports: Speed (tokens/sec), Memory Usage, and theoretical "Quantum Advantage" metrics.
    *   **Goal**: Quantify the trade-offs.

## Verification Plan

### Automated Verification
*   **Command**: `python src/cst/quantum/run_cst.py --mode demo`
    *   *Success Criteria*: Script runs without error, prints "Ambiguity Resolution" table.
*   **Command**: `python src/cst/quantum/run_cst.py --mode train --epochs 1 --batch_size 2`
    *   *Success Criteria*: Training completes 1 epoch, saves `checkpoint.pt`.

### Manual Verification
*   Inspect the "Ambiguity" output to ensure the model produces different embeddings for polysemous words (this confirms the "Contextual" part is working).
