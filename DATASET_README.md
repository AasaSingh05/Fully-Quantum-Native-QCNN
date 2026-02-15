# Custom Dataset Support for QCNN

This directory contains utilities and examples for using custom datasets with the Quantum CNN.

## Quick Start

```bash
# Train with your NPZ file
python main.py --dataset npz --path /path/to/your/data.npz

# Train with CSV
python main.py --dataset csv --path /path/to/your/data.csv

# Train with MNIST
python main.py --dataset mnist --samples 500
```

## Files Created

### Core Utilities
- **`QCNN/utils/dataset_loader.py`** - Universal dataset loader supporting NPZ, CSV, images, MNIST
- **`QCNN/utils/data_preprocessing.py`** - Preprocessing functions for quantum compatibility
- **`QCNN/config/Qconfig.py`** - Updated with dataset configuration parameters
- **`QCNN/training/Qtrainer.py`** - Updated with dataset validation

### Examples
- **`examples/load_custom_dataset.py`** - Comprehensive examples for all dataset formats

### Documentation
- **`CUSTOM_DATASET_GUIDE.md`** - Complete usage guide (see artifact)

## Supported Formats

1. **NPZ files** - NumPy archives with X and y arrays
2. **CSV files** - Tabular data with features and labels
3. **Image directories** - Folder structure with class subdirectories
4. **MNIST** - Automatic download and preprocessing
5. **NumPy arrays** - Direct array input

## Data Requirements

- **Binary classification** (2 classes)
- Features automatically resized/padded to match `n_qubits` (default: 16)
- Labels automatically converted to {-1, +1}
- Features normalized to [0, 2π] for quantum encoding

## Example Usage

```python
from QCNN.utils import load_dataset
from QCNN.config.Qconfig import QuantumNativeConfig

config = QuantumNativeConfig()

# Load any dataset format
X, y = load_dataset(
    source='my_data.npz',
    dataset_type='npz',
    n_qubits=config.n_qubits,
    image_size=config.image_size
)

# Data is now ready for training!
```

See `CUSTOM_DATASET_GUIDE.md` for complete documentation.
