# Fully Quantum-Native QCNN 
### High-Resolution Image Classification with 100% Quantum Operations

This repository implements a **fully quantum-native convolutional neural network (QCNN)** that performs encoding, convolution, pooling, and classification **entirely using quantum circuits**. Unlike hybrid models, this architecture contains no classical convolutional layers at any stage.

The project has evolved from a small 4x4 prototype to a robust framework capable of classifying **high-resolution images (e.g., 28x28 MNIST)** with over **95% accuracy** using advanced encoding strategies like **Amplitude Embedding** and **Patch-based Quanvolution**.

---

## Key Features

- **100% Quantum-Native**: Convolution, Pooling, and Classification are all implemented as differentiable quantum circuits.
- **High Performance**: Achieving **95.2% accuracy** on MNIST (0 vs 1) using only quantum operations.
- **Advanced Encoding**:
  - **Amplitude Encoding**: Maps full images into Hilbert space using only $\log_2(N)$ qubits.
  - **Patch-based Encoding**: Uses a sliding quantum filter (Quanvolution) to process large images efficiently.
- **Stable Training**: Integrated **Exponential Moving Average (EMA)**, **Learning Rate Warmup**, and **Gradient Clipping** to prevent oscillations and barren plateaus.
- **Strict Binary Classification**: Easily switch between any two classes (e.g., MNIST 3 vs 7) via simple CLI arguments.
- **Visualization Tools**: Built-in utilities for visualizing the quantum preprocessing pipeline and circuit architectures.
- **Optimized Simulation**: Support for `pennylane-lightning` and multiprocessing for accelerated training.

---

## Repository Structure

```text
QCNN/
├── config/Qconfig.py           # Centralized configuration (Architecture, LR, EMA)
├── encoding/QEncoder.py        # Amplitude, Feature Map, and Patch-based encoders
├── layers/
│   ├── QConv.py                # 2x2 weight-shared quantum kernels
│   ├── QPool.py                # Trainable unitary pooling layers
│   └── QuanvLayer.py           # Random quantum filters for patch preprocessing
├── models/QCNNModel.py         # The core PureQuantumNativeCNN implementation
├── training/Qtrainer.py        # Optimized training loop (Adam, EMA, Early Stopping)
└── utils/
    ├── dataset_loader.py       # Universal loader (MNIST, NPZ, CSV, Images)
    ├── visualize_preprocessing.py # Preprocessing visualization tool
    └── metadata_logger.py      # Experiment tracking and reproducibility
main.py                         # Unified entry point for training and evaluation
RUN_GUIDE.md                    # Quick-start guide and CLI examples
DATASET_README.md               # Technical details on data handling
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install pennylane pennylane-lightning numpy matplotlib scikit-learn seaborn
```

### 2. Run Standard Training
Classify MNIST digits 0 vs 1 using Amplitude Encoding:
```bash
python main.py --dataset mnist --classes 0 1 --samples 500 --encoding amplitude
```

### 3. Run with Patch-based Encoding
Useful for larger images or different feature extraction:
```bash
python main.py --dataset mnist --classes 3 7 --encoding patch
```

---

## Performance Benchmarks

Modern QCNN performance on MNIST (Binary Classification):

| Task | Encoding | Qubits | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **0 vs 1** | Amplitude | 8 | **95.2%** | Verified |
| **3 vs 7** | Amplitude | 8 | **91.4%** | Verified |
| **0 vs 1** | Patch | 16 | **94.8%** | Verified |

*Results obtained using Adam optimizer with EMA and LR Warmup.*

---

## Architecture Deep Dive

### Encoding Strategies
- **Amplitude Encoding**: Efficiently stores $2^n$ features in $n$ qubits. Ideal for preserving global structure with minimal hardware requirements.
- **Patch-based (Quanv)**: Applies a "Quantum Convolutional Layer" with random filters to extract local features before the main QCNN, enabling the processing of images of any size.

### Quantum Training Pipeline
We implement several stability techniques to ensure convergence in the quantum landscape:
1. **EMA (Exponential Moving Average)**: Maintains a shadow copy of weights for smoother inference.
2. **LR Warmup**: Gradually increases learning rate to avoid unstable initial gradients.
3. **Gradient Clipping**: Prevents "exploding" updates common in variational circuits.
4. **Early Stopping**: Monitors validation performance to prevent overfitting.

---

## Visualization
You can visualize how the quantum model "sees" the data:
```bash
python QCNN/utils/visualize_preprocessing.py --dataset mnist --classes 0 1
```
This generates comparisons between raw images and their quantum-encoded counterparts (e.g., amplitude maps or quanvolved patches).

---

## Reproducibility & Experiments

All experiments are seed-controlled and reproducible. Use the exact pinned
environment for matching numbers:

```bash
pip install -r requirements-lock.txt     # exact tested versions (Python 3.14)
# or, for a looser install:
pip install -r requirements.txt
```

Regenerate everything with one script:

```bash
./reproduce.sh            # full study (many QCNN trainings — slow)
./reproduce.sh --quick    # fast smoke test that exercises the whole pipeline
```

### Ablation & multi-seed study
A single harness runs the ablations across hard MNIST digit pairs and multiple
seeds, then reports **mean ± std** for accuracy / precision / recall / F1 /
ROC-AUC / PR-AUC, alongside both classical baselines (logistic, MLP) and
published **quantum-architecture** baselines (see below), all trained on the
**identical** split for a fair comparison:

```bash
python -m experiments.run_experiments \
    --datasets 0,1 3,5 4,9 5,8 --seeds 0 1 2 3 4 --samples 400 --epochs 30
# -> Results/experiments/summary.csv  (one row per dataset × config, mean ± std)
```

Ablation toggles (set in `QCNN/config/Qconfig.py` or via the runner):

| Component | Flag | Variants |
| :--- | :--- | :--- |
| Pooling | `pooling_mode` | `unitary` (proposed) · `none` · `measurement` |
| Conv entanglement | `conv_entanglement` | `full` · `one_diagonal` · `none` |
| Kernel rotations | `kernel_rotations` | `su2` (proposed) · `ry` |
| Encoding | `encoding_type` | `amplitude` · `feature_map` |

### Quantum-architecture baselines
To compare the proposed FQCNN against *existing* quantum architectures (not only
classical models), the harness also trains three well-cited quantum classifiers
on the **identical** amplitude-encoded representation, split, seed, optimiser and
epoch budget (`baselines/quantum_baselines.py`). They appear automatically as
extra rows in `summary.csv` (`baseline_cong`, `baseline_hur`, `baseline_ttn`),
each reporting its own trainable-parameter count for a like-for-like comparison:

| Row | Architecture | Reference |
| :--- | :--- | :--- |
| `baseline_cong` | Canonical QCNN (translationally-invariant conv + pooling) | Cong, Choi & Lukin, *Nature Physics* **15**, 1273 (2019) |
| `baseline_hur` | QCNN with a more expressive two-qubit conv block | Hur, Kim & Park, *EPJ Quantum Technology* **9**, 1 (2022) |
| `baseline_ttn` | Tree tensor network hierarchical classifier | Grant et al., *npj Quantum Information* **4**, 65 (2018) |

```bash
# train/evaluate the quantum baselines standalone on one MNIST pair (debugging)
python -m baselines.quantum_baselines --classes 0 1 --samples 120 --epochs 15
```

### Metrics
Every run computes the full metric suite via `QCNN/utils/metrics.py` and writes a
`metrics.json`. A single training run also saves confusion-matrix, ROC, and
precision-recall figures under `Results/Graphs/`.

### Noise robustness (NISQ)
Two free, local noise models — uniform depolarizing and a calibrated,
IBM-hardware-like multi-channel model (depolarizing + amplitude/phase damping +
readout error):

```bash
python noise_sim.py --noise-model depolarizing --classes 0 1
python noise_sim.py --noise-model realistic    --classes 0 1
```

### Optional: real quantum hardware
`experiments/hardware_run.py` runs the trained circuit on IBM Quantum's free
Open Plan. It needs `pip install pennylane-qiskit qiskit-ibm-runtime` and a free
token in `IBM_QUANTUM_TOKEN`; it skips cleanly if either is missing.

---

## Citation & Contact
**Author**: Aasa Singh Bhui  
**Project**: Fully Quantum-Native QCNN  
**GitHub**: [AasaSingh05](https://github.com/AasaSingh05)

If you use this work, please cite the repository:
```text
https://github.com/AasaSingh05/Fully-Quantum-Native-QCNN
```