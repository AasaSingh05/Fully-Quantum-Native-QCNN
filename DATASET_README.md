# Custom Dataset Support for QCNN

This document contains a comprehensive guide for using custom datasets with the Quantum CNN, including loading, preprocessing, and integration.

---

# Custom Dataset Usage Guide for QCNN

This guide explains how to train the Quantum CNN with your own datasets.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Dataset Formats](#supported-dataset-formats)
3. [Data Requirements](#data-requirements)
4. [Usage Examples](#usage-examples)
5. [Preprocessing Your Data](#preprocessing-your-data)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using Command Line

```bash
# Train with synthetic dataset (default)
python main.py --dataset synthetic --samples 100

# Train with your NPZ file
python main.py --dataset npz --path /path/to/your/data.npz

# Train with CSV file
python main.py --dataset csv --path /path/to/your/data.csv

# Train with MNIST subset
python main.py --dataset mnist --samples 500
```

### Using Python API

```python
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from QCNN.utils import load_dataset
from sklearn.model_selection import train_test_split

# Load your dataset
config = QuantumNativeConfig()
X, y = load_dataset(
    source='my_data.npz',
    dataset_type='npz',
    n_qubits=config.n_qubits,
    image_size=config.image_size
)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = PureQuantumNativeCNN(config)
trainer = QuantumNativeTrainer(learning_rate=0.005)
trained_model = trainer.train_pure_quantum_cnn(model, X_train, y_train, X_test, y_test)
```

---

## Supported Dataset Formats

### 1. NPZ Files (NumPy Archive)

**Format Requirements:**
- File extension: `.npz`
- Must contain arrays named `X` (features) and `y` (labels)
- Alternative names: `data`/`features`/`images` for X, `labels`/`targets` for y

**Example:**
```python
import numpy as np

# Create dataset
X = np.random.randn(1000, 16)  # 1000 samples, 16 features
y = np.random.choice([0, 1], 1000)  # Binary labels

# Save as NPZ
np.savez('my_dataset.npz', X=X, y=y)

# Load with QCNN
from QCNN.utils import load_dataset
X_processed, y_processed = load_dataset(
    source='my_dataset.npz',
    dataset_type='npz',
    n_qubits=16
)
```

### 2. CSV Files

**Format Requirements:**
- File extension: `.csv`
- Features in columns, samples in rows
- One column for labels (default: last column)
- Optional header row

**Example:**
```python
# Load CSV with label in last column
X, y = load_dataset(
    source='data.csv',
    dataset_type='csv',
    n_qubits=16,
    label_column=-1  # Last column
)

# Load CSV with named label column
X, y = load_dataset(
    source='data.csv',
    dataset_type='csv',
    n_qubits=16,
    label_column='target'  # Column name
)
```

### 3. Image Directories

**Format Requirements:**
- Directory structure: `root_dir/class_0/*.png` and `root_dir/class_1/*.png`
- Exactly 2 class subdirectories for binary classification
- Supported formats: PNG, JPG, JPEG

**Example:**
```python
# Directory structure:
# images/
#   ├── cats/
#   │   ├── cat1.png
#   │   └── cat2.png
#   └── dogs/
#       ├── dog1.png
#       └── dog2.png

X, y = load_dataset(
    source='images/',
    dataset_type='images',
    n_qubits=16,
    image_size=4,  # Resize to 4x4
    max_samples=100  # Limit per class
)
```

### 4. MNIST Dataset

**Format Requirements:**
- Automatically downloaded via scikit-learn
- Requires: `pip install scikit-learn`

**Example:**
```python
# Load MNIST digits 0 vs 1
X, y = load_dataset(
    source='mnist',
    dataset_type='mnist',
    n_qubits=16,
    image_size=4,
    n_samples=1000,
    classes=(0, 1)  # Binary classification
)
```

### 5. Direct NumPy Arrays

**Example:**
```python
# Your custom data
X_raw = np.random.randn(500, 28, 28)  # 500 images, 28x28
y_raw = np.random.choice([0, 1], 500)

# Load as tuple
X, y = load_dataset(
    source=(X_raw, y_raw),
    dataset_type='array',
    n_qubits=16,
    image_size=4
)
```

---

## Data Requirements

### Feature Requirements

> [!IMPORTANT]
> **Quantum Circuit Constraints**
> - Number of features must match `n_qubits` (default: 16)
> - Features are encoded as rotation angles in [0, 2π]
> - Data is automatically normalized during loading

**Default Configuration:**
- `n_qubits = 16` (4×4 image)
- `image_size = 4`
- Features per sample: 16

### Label Requirements

> [!WARNING]
> **Binary Classification Only**
> - Exactly 2 classes required
> - Labels automatically converted to {-1, +1}
> - Original labels can be any format (0/1, cat/dog, etc.)

---

## Usage Examples

### Example 1: Load Your Own NPZ File

```python
from QCNN.utils import load_dataset
from QCNN.config.Qconfig import QuantumNativeConfig

config = QuantumNativeConfig()

# Your NPZ file with any feature count
X, y = load_dataset(
    source='my_research_data.npz',
    dataset_type='npz',
    n_qubits=config.n_qubits,
    image_size=config.image_size,
    normalization='minmax'  # Options: 'minmax', 'standard', 'robust'
)

print(f"Loaded: {X.shape}, Labels: {np.unique(y)}")
# Output: Loaded: (1000, 16), Labels: [-1  1]
```

### Example 2: Load CSV with Custom Preprocessing

```python
# CSV with 100 features, but we need 16
X, y = load_dataset(
    source='high_dim_data.csv',
    dataset_type='csv',
    n_qubits=16,  # Will truncate to first 16 features
    normalization='standard',  # Z-score normalization
    label_column='diagnosis'
)
```

### Example 3: Load Images and Train

```python
from QCNN.utils import load_dataset
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from sklearn.model_selection import train_test_split

# Load images
config = QuantumNativeConfig()
X, y = load_dataset(
    source='my_images/',
    dataset_type='images',
    n_qubits=16,
    image_size=4
)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = PureQuantumNativeCNN(config)
trainer = QuantumNativeTrainer()
trained_model = trainer.train_pure_quantum_cnn(
    model, X_train, y_train, X_test, y_test
)
```

### Example 4: Custom Preprocessing Pipeline

```python
from QCNN.utils.data_preprocessing import (
    normalize_to_quantum_range,
    encode_labels,
    flatten_and_pad
)

# Your raw data
X_raw = np.random.randn(1000, 50)  # 50 features
y_raw = np.array(['positive', 'negative'] * 500)

# Manual preprocessing
X_flat = flatten_and_pad(X_raw, target_features=16)  # Truncate to 16
X_normalized = normalize_to_quantum_range(X_flat, method='minmax')
y_encoded = encode_labels(y_raw, encoding='binary')

# Now ready for training
print(X_normalized.shape)  # (1000, 16)
print(np.unique(y_encoded))  # [-1  1]
```

---

## Preprocessing Your Data

### Automatic Preprocessing

The `load_dataset()` function automatically:

1. **Resizes** images to match `image_size × image_size`
2. **Flattens** multi-dimensional data
3. **Pads or truncates** to match `n_qubits`
4. **Normalizes** features to [0, 2π] range
5. **Encodes** labels to {-1, +1}

### Manual Preprocessing

For fine-grained control:

```python
from QCNN.utils.data_preprocessing import preprocess_for_quantum

# Your data
X_raw = load_your_data()  # Any shape
y_raw = load_your_labels()  # Any format

# Preprocess
X_processed, y_processed = preprocess_for_quantum(
    X_raw, y_raw,
    n_qubits=16,
    image_size=4,
    normalization='standard'  # 'minmax', 'standard', or 'robust'
)
```

### Normalization Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `minmax` | Scale to [0, 2π] linearly | Most datasets (default) |
| `standard` | Z-score, clip to ±3σ | Data with outliers |
| `robust` | Median/IQR scaling | Heavy outliers |

---

## Troubleshooting

### Error: "Feature count doesn't match n_qubits"

**Problem:** Your data has wrong number of features.

**Solution:**
```python
# Let load_dataset handle it automatically
X, y = load_dataset(
    source=your_data,
    n_qubits=16,  # Specify target
    image_size=4  # Will resize/pad as needed
)
```

### Error: "Labels must be {-1, +1}"

**Problem:** Labels not in correct format.

**Solution:**
```python
from QCNN.utils.data_preprocessing import encode_labels

# Convert any labels to {-1, +1}
y_encoded = encode_labels(y_raw, encoding='binary')
```

### Warning: "Features should be in [0, 2π] range"

**Problem:** Data not normalized for quantum encoding.

**Solution:**
```python
from QCNN.utils.data_preprocessing import normalize_to_quantum_range

X_normalized = normalize_to_quantum_range(X, method='minmax')
```

### Low Accuracy with Custom Data

**Possible causes:**

1. **Data not suitable for quantum advantage**
   - Try synthetic dataset first to verify setup
   - Quantum CNNs excel at spatially-structured patterns

2. **Insufficient preprocessing**
   - Ensure normalization is appropriate
   - Check class balance

3. **Hyperparameter tuning needed**
   ```python
   # Adjust in Qconfig.py
   config.learning_rate = 0.01  # Try different values
   config.n_epochs = 150  # More epochs
   config.batch_size = 8  # Smaller batches
   ```

---

## File Structure Reference

- **`QCNN/utils/dataset_loader.py`** - Universal dataset loader
- **`QCNN/utils/data_preprocessing.py`** - Preprocessing utilities
- **`examples/load_custom_dataset.py`** - Comprehensive examples
- **`main.py`** - Training entry point with CLI support
