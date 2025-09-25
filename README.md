# Quantum-Native QCNN

This repository contains a fully quantum-native convolutional neural network (QCNN) that implements encoding, convolution, pooling, and classification purely with quantum circuits on a 4×4 grid of qubits (16 wires) using PennyLane. The model uses shared quantum kernels over 2×2 windows, unitary pooling that halves the active qubits per stage, and a shallow variational head that reads an expectation on the final qubit.

## What this is

The core idea is to replicate CNN-like structure entirely with quantum operations: angle-based feature maps for data encoding, weight-shared quantum “kernels” sliding over 2×2 patches, trainable unitary pooling to downsample, and a final measurement producing a scalar readout. Everything after the dataset loader runs as a quantum circuit; there’s no classical convolution or pooling path here.

## Repository layout

- **main.py**: entry point for dataset generation/loading, training, evaluation, and plotting; writes results to Results/
- **QCNN/config/Qconfig.py**: configuration, including image_size=4, n_qubits=16, encoding_type='feature_map', and training hyperparameters; pooling_reduction defaults to 0.5 so the register halves each stage.
- **QCNN/encoding/QEncoder.py**: PureQuantumEncoder with amplitude_encoding and a symmetric quantum_feature_map (RZ(πx), RY(πx/2) with x∈[−1,1]) plus lightweight entanglement.
- **QCNN/layers/QConv.py**: QuantumNativeConvolution with a shared 2×2 kernel per layer, using per-qubit RY/RZ and local entanglement across the 2×2 window; no data-dependent rescaling, autograd-safe parameter handling.
- **QCNN/layers/QPool.py**: QuantumNativePooling with unitary pooling via CRY/CRZ from discard→keep and a simple make_pairing helper; model drops discarded wires after pooling.
- **QCNN/models/QCNNModel.py**: PureQuantumNativeCNN integrating encoder, sliding 2×2 windows over the active layout per stage, halving wires via pooling, and a shallow head; returns ⟨Z⟩ on the readout wire.
- **QCNN/training/Qtrainer.py**: QuantumNativeTrainer using Adam with parameter-shift gradients; supports original MSE on ⟨Z⟩ and an optional BCE path on p=(1−⟨Z⟩)/2.
- **QCNN/utils/dataset_generator.py**: synthetic binary dataset generator returning X and labels in {−1,1}; cached at Results/quantum_dataset.npz.
- **Visualization utilities (optional)**: visualize_all_circuits.py renders circuits for encoder/conv/pool/head/full-forward to Results/circuits/, and visualize_4qubit_circuits.py renders a minimal 4-qubit pipeline to Results/4Circuit/.

## Setup

- Python dependencies: PennyLane, a simulator backend (default.qubit or lightning.qubit), NumPy, Matplotlib, scikit-learn.
- Install basics:
  - `pip install pennylane pennylane-lightning numpy matplotlib scikit-learn`

## Running

- From the repo root:
  - `python main.py`
- On first run, a dataset is generated and cached under `Results/quantum_dataset.npz`; subsequent runs reuse it unless the file is deleted.
- Training writes:
  - `Results/Weights/quantum_model_params.npz` (best checkpoint),
  - `Results/Graphs/quantum_training_results.png` (loss and test accuracy curves),
  - `quantum_training_log.txt` (batch diagnostics).

## Model details

- **Encoding**: angle-based feature map on 16 qubits; data is clipped to [−1,1], then mapped with RZ(πx) and RY(πx/2), with sparse entanglement to keep circuits shallow and trainable; amplitude encoding remains an option but is not default for 4×4.
- **Convolution**: each stage uses a single trainable kernel with shape (4, depth, 2) for [RY,RZ] per qubit per depth; the kernel is applied to every 2×2 window discovered on the current active grid (weight sharing).
- **Pooling**: pairs wires as (keep, discard) and applies CRY/CRZ from discard→keep, then keeps only the designated wires for the next stage; no per-input normalization of pooling angles.
- **Head**: one or two single-qubit rotations plus a light entangler when two wires remain; readout via expval(Z) on the first active wire; predictions map ⟨Z⟩ to {-1,1} for evaluation.
- **Training**: Adam optimizer on parameter-shift gradients; default loss is MSE on ⟨Z⟩ vs labels in {−1,1}; an optional BCE path on p=(1−⟨Z⟩)/2 is available to align with probabilistic classification.

## Notes on stability

- To reduce gradient variance, batch_size=8 and learning_rate=0.005 are reasonable starting points; increase batch size or reduce LR if the loss curve is excessively spiky.
- Circuit depth is intentionally shallow (conv depth=1) to keep optimization stable on simulators and better reflect near-term devices.
- If shots are enabled later, expect additional stochasticity; keeping shots=None (analytic) is recommended during development.

## Visualizing circuits

- Full 16-qubit circuits and module-specific diagrams:
  - `python visualize_all_circuits.py` (saves to `Results/circuits/`)
- Minimal 4-qubit end-to-end circuits:
  - `python visualize_4qubit_circuits.py` (saves to `Results/4Circuit/`)

## Reproducibility

- The config sets seed=42 for consistent parameter initialization; dataset generation also uses a fixed split unless the cache is removed and the random_state is changed in main.py.

## Common issues

- “setting an array element with a sequence”: typically occurs if parameters are forcibly cast to float before autograd finishes tracing; QConv and QPool avoid dtype forcing and reshape parameters safely with np.asarray, which resolves it.
- AttributeError: make_pairing: ensure QCNN/layers/QPool.py includes QuantumNativePooling.make_pairing and that QCNNModel uses it to pair wires before pooling.
- Perfect test accuracy: verify no data leakage by regenerating the dataset (delete Results/quantum_dataset.npz), checking stratified splits, and validating on fresh seeds; also sanity-check the dataset generator for trivial label rules.

## Why this is quantum-native

The entire CNN stack—encoding, convolution, pooling, classification—is composed of quantum gates acting on qubits, with weight sharing and downsampling implemented as unitary transformations rather than classical ops; the only classical components are the optimizer and the loss computation fed by quantum measurements.
