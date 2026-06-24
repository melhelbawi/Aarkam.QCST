# 🤝 Contributing to Quantum CST

Thank you for your interest in contributing to **Contextual Spectrum Tokenization (CST)**. We are at the intersection of quantum information theory and transformer-based NLP, and your expertise—whether in physics, machine learning, or software engineering—is invaluable.

---
## License

This project is released under the **CST / QCST Dual License**.
Commercial use is strictly prohibited without explicit written permission.

## 🚀 Getting Started

### 1. Environment Configuration
We utilize a strictly decoupled architecture. You can contribute to either the classical or quantum modules independently.

```bash
# Recommended: Python 3.10+
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install Quantum-specific dev dependencies
pip install -r src/cst/quantum/requirements.txt
pip install pytest black ruff  # Dev tools
```

### 2. Verification Baseline
Ensure all tests pass before starting development:
```bash
python src/cst/quantum/tests/test_quantum_imports.py
```

---

## 🛠️ Developer Workflow

### Feature Branches
We follow a standard Git Flow. Please create descriptive branch names:
- `feat/quantum-attention-heads`
- `fix/cache-invalidation-logic`
- `perf/vqc-gate-pruning`

### Code Excellence Standards
- **Strict Typing**: All new functions must include Python type hints.
- **Zero Decoupling Breaches**: Quantum modules (`src/cst/quantum/`) must **never** import from classical modules.
- **Docstrings**: Use the Google Python Style Guide for docstrings.

---

## ⚛️ Quantum Development Guidelines

### 1. Variational Circuit Design
When modifying the `QuantumInformationFuser` or adding new circuits:
- **Barren Plateaus**: Be mindful of parameter initialization; avoid all-zero starts.
- **Qubit Mapping**: Document the semantic mapping of each qubit in the circuit docstring.

### 2. PennyLane Compatibility
Target compatibility with the `default.qubit` simulator for all unit tests. Hardware-specific backends (IBM, IonQ) should be kept optional.

---

## 📝 Pull Request Template

When submitting a PR, please include the following:

```markdown
## Overview
Briefly describe the change and the problem it solves.

## Technical Details
- [ ] Quantum-Classical decoupling maintained (no classical imports).
- [ ] All 8 core tests passing.
- [ ] Type hints and Google-style docstrings added.

## Performance Impact
- Parameter Efficiency Delta: (e.g., +5%)
- Inference Latency Change: (e.g., -10ms)
```

---

## ⚖️ License
By contributing, you agree that your contributions will be licensed under the **CST / QCST Dual License**.

---

**Maintaining Research Integrity**: As this is a research-grade project, please cite any papers or mathematical foundations used in your logic within the code comments.

**Lead Maintainer**: Mohamed Elhelbawi
