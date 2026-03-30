# QCNN Execution Guide 🚀

This guide explains how to run the Fully Quantum-Native QCNN for various tasks, from standard training to advanced preprocessing visualization.

---

## 🏗️ Environment Setup

Ensure you have the required packages installed:
```bash
pip install pennylane pennylane-lightning numpy matplotlib scikit-learn seaborn pillow
```

---

## 🏃 Running Training (main.py)

The `main.py` script is the central entry point. It supports a variety of datasets and encoding strategies.

### 1. MNIST Classification (Standard)
Classify digits 0 and 1 using **Amplitude Encoding** (8 qubits for 256 pixels):
```bash
python main.py --dataset mnist --classes 0 1 --samples 1000 --encoding amplitude
```

### 2. High-Resolution Classification (Patch-based)
Used for larger images or when local feature extraction is critical. This uses **16 qubits** by default for the final processing:
```bash
python main.py --dataset mnist --classes 3 7 --samples 500 --encoding patch
```

### 3. Custom Datasets
- **NPZ File**: `python main.py --dataset npz --path datasets/my_data.npz`
- **CSV File**: `python main.py --dataset csv --path datasets/my_data.csv`
- **Image Directory**: `python main.py --dataset images --path datasets/my_images/` (Ensure folders `class_0` and `class_1` exist)

---

## 🎨 Visualization Tools

### Quantum Preprocessing Visualizer
See how the model transforms raw images into quantum-compatible states:
```bash
python QCNN/utils/visualize_preprocessing.py
```
*Outputs are saved to `Results/preprocessing_images/`.*

---

## ⚙️ Advanced CLI Arguments

| Argument | Options | Description |
| :--- | :--- | :--- |
| `--dataset` | `mnist`, `npz`, `csv`, `images`, `synthetic` | Source of training data |
| `--classes` | `INT INT` | Two classes for binary classification (e.g., `0 1`) |
| `--encoding` | `amplitude`, `patch`, `feature_map` | How data is mapped to qubits |
| `--samples` | `INT` | Max samples to use (for faster testing) |
| `--learning-rate`| `FLOAT` | Initial learning rate (default: 0.02) |
| `--epochs` | `INT` | Number of training epochs (default: 50) |
| `--batch-size` | `INT` | Training batch size (default: 32) |
| `--use-mse` | (flag) | Use Mean Squared Error instead of BCE |
| `--no-profile` | (flag) | Disable performance profiling |

---

## 📊 Monitoring Results

After running `main.py`, results are automatically organized in the `Results/` directory:

- **`Results/Graphs/`**: Training curves, Confusion Matrices, ROC curves, and Precision-Recall plots.
- **`Results/Weights/`**: Saved model parameters (`.npy` files).
- **`Results/metadata.json`**: Comprehensive log of the experiment configuration.
- **`training_summary.txt`**: A concise per-epoch summary of loss and accuracy.
- **`training_log.txt`**: Detailed terminal output.
