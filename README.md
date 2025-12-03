# Quantum-Native QCNN  
A Fully Quantum-Convolutional Neural Network Built with PennyLane

This repository implements a **fully quantum-native convolutional neural network (QCNN)** that performs **encoding, convolution, pooling, and classification entirely using quantum circuits**, with no classical convolutional operations at any stage.

The current prototype operates on **4×4 images mapped onto a 16-qubit grid**, using translationally shared quantum kernels, unitary pooling layers, and a variational quantum head. All training is performed using differentiable quantum circuits and parameter-shift gradients.

This project is part of a broader research effort toward **quantum-native deep learning architectures** suitable for near-term quantum devices.

---

## 1. What This Project Implements

The QCNN reproduces the structure of classical CNNs using **only quantum operations**:

- **Feature Encoding** via angle-based quantum feature maps  
- **Quantum Convolution** using weight-shared 2×2 local quantum kernels  
- **Quantum Pooling** via trainable 2-qubit unitary reductions  
- **Quantum Classification Head** producing a scalar prediction ⟨Z⟩  
- **End-to-end Differentiability** through PennyLane's autodiff and parameter-shift rule

There is **no hybrid classical convolution path**.  
Every layer (conv, pool, head) is a quantum circuit.

---

## 2. Repository Structure

```
QCNN/
│
├── config/Qconfig.py              # global configuration (qubits, layers, LR, encoding)
├── encoding/QEncoder.py           # angle-based encoding + optional amplitude encoding
├── layers/QConv.py                # 2×2 shared quantum convolution kernel
├── layers/QPool.py                # unitary quantum pooling (CRY/CRZ)
├── models/QCNNModel.py            # full QCNN architecture
├── training/Qtrainer.py           # quantum training loop (Adam + param-shift)
└── utils/
     ├── dataset_generator.py      # reproducible synthetic dataset
     └── metadata_logger.py        # experiment metadata for reproducibility

run_classical_baseline.py          # standalone classical CNN baseline
main.py                            # QCNN training/evaluation entry point
README.md                          # this file
```

---

## 3. Setup

Install the core dependencies:

```bash
pip install pennylane pennylane-lightning numpy matplotlib scikit-learn tensorflow
```

(Optional) Use Lightning for faster simulation:

```bash
pip install pennylane-lightning
```

---

## 4. Running the Quantum Model

Run the full QCNN experiment:

```bash
python main.py
```

On the first execution:

- A clean dataset is generated under:  
  `Results/quantum_dataset_clean_seed42.npz`

- Training artifacts are saved under:
  ```
  Results/Weights/
  Results/Graphs/
  Results/metadata.json
  quantum_training_log.txt
  ```

Outputs include:
- Training loss curve  
- Test accuracy curve  
- Confusion matrix  
- Best model weights  
- Experiment metadata (Python version, package versions, config)

---

## 5. Running the Classical Baseline (Recommended)

To compare with a classical CNN baseline:

```bash
python run_classical_baseline.py
```

This script:

- Loads the same dataset used by the QCNN  
- Trains a small 4×4 classical CNN  
- Saves plots to:
  ```
  Results/Baselines/classical_cnn_results.png
  ```
- Prints baseline accuracy and confusion matrix

This ensures the QCNN is not solving a trivial dataset.

---

## 6. QCNN Architecture Details

### Encoding
- Angle-based feature map on all 16 qubits  
- Uses:
  - `RZ(πx)`  
  - `RY(πx/2)`
- Sparse entanglement for trainability  
- Inputs clipped to [−1, 1]

### Quantum Convolution
- Shared 2×2 quantum kernel applied across the grid  
- Kernel shape:
  ```
  (4 qubits per window) × (depth) × (2 learnable angles)
  ```
- RY/RZ + local entanglement  
- Enforces translational invariance

### Quantum Pooling
- Trainable unitary pooling  
- CRY/CRZ from discarded → kept qubit  
- Removes half the qubits each stage  
- No measurement until the final layer

### Quantum Head
- Single-qubit variational readout  
- Optional entanglement when >1 qubit remains  
- Final output: ⟨Z⟩ ∈ [−1, 1]

### Training
- Optimizer: Adam  
- Gradients: parameter-shift rule  
- Loss options:
  - MSE on ⟨Z⟩  
  - BCE on p = (1 − ⟨Z⟩)/2

---

## 7. Reproducibility

- Global seeds for numpy, random, PennyLane  
- Clean dataset with fixed seed  
- Metadata logged in `Results/metadata.json`:
  - configuration  
  - Python version  
  - package versions  
  - random seed  

This ensures exact reproducibility of experiments.

---

## 8. Visualizing Circuits

To generate full QCNN circuit diagrams:

```bash
python visualize_all_circuits.py
```

Saved to:

```
Results/circuits/
```

Minimal 4-qubit demonstration circuits:

```bash
python visualize_4qubit_circuits.py
```

Saved to:

```
Results/4Circuit/
```

---

## 9. Limitations

- Dataset is synthetic and extremely small (4×4)  
- No noise-model simulation yet  
- No real quantum hardware execution  
- QCNN uses all 16 qubits at once (not scalable yet)  
- No qubit-reuse or patch-based encoding  
- No gate-count or circuit-depth profiling  
- No gradient variance (barren plateau) analysis  
- Perfect accuracy may indicate dataset simplicity  
- No hybrid (classical–quantum) baselines yet  

---

## 10. Planned Future Work

- Patch-based QCNN with qubit reuse (scaling to 8×8, 16×16)  
- Noise-aware QCNN training (`default.mixed`)  
- Running pooled circuits on IBM/IonQ hardware  
- Computing circuit depth and gate counts  
- Gradient variance diagnostics  
- Hybrid QCNN/CNN baseline comparisons  
- Real datasets (downsampled MNIST / Fashion-MNIST)  
- Preparation of an arXiv-ready research paper  

---

## 11. Why This Architecture Is Quantum-Native

- Convolution implemented via quantum kernels  
- Pooling done via entangling unitaries  
- Downsampling in quantum state space (no classical ops)  
- Classification via quantum measurement  
- Entire model is a single differentiable quantum circuit  
- No classical CNN operations are used at any stage  

The only classical components:
- optimizer  
- loss function  
- data loading  

---

## 12. Citation

```
Author: Aasa Singh Bhui  
Project: Fully Quantum-Native QCNN (2025)  
Repository: https://github.com/AasaSingh05/AICourseProject
```

---

## 13. Contact

Aasa Singh Bhui
VIT Vellore  
Email: your.email@example.com  
GitHub: https://github.com/your_username
