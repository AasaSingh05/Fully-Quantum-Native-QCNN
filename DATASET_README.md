# Advanced Dataset Guide for QCNN

This guide provides technical details on how the Fully Quantum-Native QCNN handles data, including modern binary classification modes, diverse encoding strategies, and preprocessing pipelines.

---

## Strict Binary Classification

The model now operates in a **Strict Binary Classification** mode. Instead of "one-vs-rest", it filters the entire dataset to only include two specific classes of interest.

### Configuration
You can specify the target classes in `QCNN/config/Qconfig.py` or via the CLI:
```bash
# Compare MNIST digits 3 and 7
python main.py --dataset mnist --classes 3 7
```

- **Internals**: The `dataset_loader` filters the raw labels (e.g., 0-9) to only include users' chosen pair.
- **Label Mapping**: The chosen classes are mapped to **+1** (first class) and **-1** (second class) for quantum state compatibility.

---

## Quantum Encoding Strategies

The QCNN supports three distinct ways of mapping classical data into quantum states:

### 1. Amplitude Encoding (`--encoding amplitude`)
- **Efficiency**: Stores $2^N$ features using only $N$ qubits. 
- **Usage**: Perfect for 16x16 or 32x32 images. It allows a 256-pixel image to be encoded into just **8 qubits**.
- **Preprocessing**: Data is flattened and padded to the nearest power of 2, then normalized such that the vector has a unit norm (represented by state amplitudes).

### 2. Patch-based Quanvolution (`--encoding patch`)
- **Mechanism**: Slides a $k \times k$ quantum kernel (random filter) across the image to extract features.
- **Workflow**: 
  1. Image is divided into patches.
  2. Each patch is processed by a small quantum circuit.
  3. The resulting "feature map" is then fed into the main QCNN.
- **Scaling**: Enables processing of very large images by downsampling them into a quantum-readable format while preserving local spatial correlations.

### 3. Feature Map (Angle Encoding) (`--encoding feature_map`)
- **Efficiency**: 1 qubit per feature.
- **Usage**: High-precision encoding for small (e.g., 4x4) images.
- **Preprocessing**: Each pixel is mapped to a rotation angle ($RX/RY$) on its own qubit.

---

## Preprocessing Pipeline

The `load_dataset()` function in `QCNN/utils/dataset_loader.py` follows this sequence:

1. **Loading**: Reads from NPZ, CSV, IDX, or Image directories.
2. **Filtering**: Selects only the user-specified binary classes.
3. **Resizing**: Standardizes image dimensions (e.g., to 16x16 or 32x32).
4. **Normalization**:
   - `minmax`: Scales all features to $[0, 2\pi]$.
   - `standard`: Z-score normalization (centered around 0 with unit variance).
   - `robust`: Uses median and IQR to handle extreme outliers.
5. **Quantum Casting**: Converts data into the specific format required by the chosen `encoding_type`.

---

## Quick Test: Visualization

To verify your dataset and see how it is transformed before hitting the quantum circuit:
```bash
python QCNN/utils/visualize_preprocessing.py --dataset mnist --classes 0 1 --encoding amplitude
```
This tool saves visual comparisons to `Results/Preprocessing_Visuals/`.

---

## Supported Formats Summary

| Type | Format | Notes |
| :--- | :--- | :--- |
| **IDX** | MNIST-style | Standard IDX-UBYTE files |
| **NPZ** | NumPy | Must contain 'X' and 'y' keys |
| **CSV** | Text | Last column assumed as label |
| **Images**| PNG/JPG | Class-based folder structure |
| **MNIST**| Scikit-learn| Auto-downloads if not found |
