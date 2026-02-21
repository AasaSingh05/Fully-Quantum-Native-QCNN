import numpy as np
import os
from typing import Tuple, Optional, Union
import struct
from pathlib import Path
from .data_preprocessing import preprocess_for_quantum


def load_npz_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from NPZ file.
    
    Args:
        filepath: Path to .npz file containing 'X' and 'y' arrays
    
    Returns:
        (X, y) tuple
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = np.load(filepath)
    
    # Try common key names
    X_keys = ['X', 'x', 'data', 'features', 'images']
    y_keys = ['y', 'Y', 'labels', 'targets']
    
    X = None
    y = None
    
    for key in X_keys:
        if key in data:
            X = data[key]
            break
    
    for key in y_keys:
        if key in data:
            y = data[key]
            break
    
    if X is None or y is None:
        raise ValueError(f"Could not find X and y in NPZ file. Available keys: {list(data.keys())}")
    
    return X, y


def load_csv_dataset(filepath: str, 
                     label_column: Union[int, str] = -1,
                     has_header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to CSV file
        label_column: Column index or name for labels (default: last column)
        has_header: Whether CSV has header row
    
    Returns:
        (X, y) tuple
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    try:
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        if isinstance(label_column, str):
            y = df[label_column].values
            X = df.drop(columns=[label_column]).values
        else:
            y = df.iloc[:, label_column].values
            X = df.drop(df.columns[label_column], axis=1).values
        
        return X, y
    
    except ImportError:
        # Fallback to numpy
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1 if has_header else 0)
        
        if isinstance(label_column, int):
            if label_column == -1:
                X = data[:, :-1]
                y = data[:, -1]
            else:
                y = data[:, label_column]
                X = np.delete(data, label_column, axis=1)
        else:
            raise ValueError("String label_column requires pandas. Install with: pip install pandas")
        
        return X, y


def load_image_directory(directory: str,
                        image_size: Tuple[int, int] = (4, 4),
                        max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from directory structure: directory/class_name/*.png
    
    Args:
        directory: Root directory containing class subdirectories
        image_size: Target size for images
        max_samples: Maximum samples to load per class
    
    Returns:
        (X, y) tuple
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required for image loading. Install with: pip install pillow")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    X_list = []
    y_list = []
    
    class_dirs = [d for d in Path(directory).iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])
    
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 class directories for binary classification, found {len(class_names)}")
    
    for class_idx, class_name in enumerate(class_names):
        class_path = Path(directory) / class_name
        image_files = list(class_path.glob('*.png')) + list(class_path.glob('*.jpg')) + \
                     list(class_path.glob('*.jpeg'))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        for img_file in image_files:
            img = Image.open(img_file).convert('L')  # Grayscale
            img = img.resize(image_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            X_list.append(img_array.flatten())
            y_list.append(class_idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y


def load_idx_dataset(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST dataset from IDX-UBYTE files.
    
    Args:
        images_path: Path to the *-images-idx3-ubyte file
        labels_path: Path to the *-labels-idx1-ubyte file
        
    Returns:
        (X, y) tuple
    """
    print(f"Loading IDX dataset from {os.path.basename(images_path)} and {os.path.basename(labels_path)}...")
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def load_mnist_subset(n_samples: int = 1000,
                     classes: Tuple[int, int] = (0, 1),
                     flatten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a subset of MNIST dataset for binary classification.
    
    Args:
        n_samples: Total number of samples to load
        classes: Two classes to use for binary classification
        flatten: Whether to flatten images
    
    Returns:
        (X, y) tuple
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError("scikit-learn required for MNIST. Install with: pip install scikit-learn")
    
    print(f"Loading MNIST subset (classes {classes[0]} and {classes[1]})...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Filter for binary classification
    mask = (y == classes[0]) | (y == classes[1])
    X = X[mask]
    y = y[mask]
    
    # Limit samples
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Convert to numpy array if needed
    if hasattr(X, 'values'):
        X = X.values
    
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    
    if not flatten and X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    
    return X, y


def load_dataset(source: Union[str, Tuple[np.ndarray, np.ndarray]],
                dataset_type: str = 'auto',
                n_qubits: int = 16,
                image_size: Optional[int] = 4,
                normalization: str = 'minmax',
                encoding_type: str = 'feature_map',
                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal dataset loader with automatic preprocessing.
    
    Args:
        source: Dataset source (file path, directory, or (X, y) tuple)
        dataset_type: Type of dataset ('auto', 'npz', 'csv', 'images', 'mnist', 'array')
        n_qubits: Number of qubits in quantum circuit
        image_size: Target image size (if applicable)
        normalization: Normalization method
        encoding_type: 'feature_map', 'amplitude', or 'patch'
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        Preprocessed (X, y) ready for quantum training
    """
    # Handle direct array input
    X = None
    y = None
    if isinstance(source, tuple) and len(source) == 2:
        X, y = source
        dataset_type = 'array'
    
    # Auto-detect dataset type
    elif dataset_type == 'auto':
        if isinstance(source, str):
            if source.endswith('.npz'):
                dataset_type = 'npz'
            elif source.endswith('.csv'):
                dataset_type = 'csv'
            elif os.path.isdir(source):
                # Check if it's an images directory or an MNIST IDX directory
                files = os.listdir(source)
                if any('idx3' in f and 'images' in f for f in files):
                    dataset_type = 'idx'
                else:
                    dataset_type = 'images'
            else:
                raise ValueError(f"Cannot auto-detect dataset type for: {source}")
        else:
            raise ValueError("Auto-detection requires string path")
    
    print(f"Loading {dataset_type} dataset from source...")
    binary_classes = kwargs.get('classes', (0, 1))
    
    # Load based on type
    if dataset_type == 'idx':
        # If source is a directory, look for standard MNIST files
        if os.path.isdir(source):
            files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
            
            # Try to find 'train' first, then 't10k', otherwise any matching pair
            prefixes = ['train', 't10k', '']
            images_file = None
            labels_file = None
            
            for pref in prefixes:
                images_file = next((f for f in files if pref in f and 'images' in f and 'idx3' in f), None)
                labels_file = next((f for f in files if pref in f and 'labels' in f and 'idx1' in f), None)
                if images_file and labels_file:
                    break
            
            if not images_file or not labels_file:
                raise FileNotFoundError(f"Could not find matching MNIST IDX files in {source}")
                
            X, y = load_idx_dataset(os.path.join(source, images_file), 
                                  os.path.join(source, labels_file))
        else:
            # Assume source is images path, and look for labels path in kwargs
            labels_path = kwargs.get('labels_path')
            if not labels_path:
                # If source ends in -images-idx3-ubyte, try to find -labels-idx1-ubyte
                labels_path = source.replace('images-idx3', 'labels-idx1')
                if not os.path.exists(labels_path):
                    raise ValueError("labels_path must be provided for IDX datasets if source is just a file")
            
            X, y = load_idx_dataset(source, labels_path)
            
        # Filter for binary classification
        print(f"Filtering for binary classes: {binary_classes}")
        mask = (y == binary_classes[0]) | (y == binary_classes[1])
        X = X[mask]
        y = y[mask]
    
    elif dataset_type == 'npz':
        X, y = load_npz_dataset(source)
    
    elif dataset_type == 'csv':
        X, y = load_csv_dataset(source, **kwargs)
    
    elif dataset_type == 'images':
        target_size = (image_size, image_size) if image_size else (4, 4)
        X, y = load_image_directory(source, image_size=target_size, **kwargs)
    
    elif dataset_type == 'mnist':
        X, y = load_mnist_subset(**kwargs)
    
    elif dataset_type == 'array' and X is None:
        X, y = source
    
    elif dataset_type == 'array':
        # Already loaded
        pass
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1:]} features")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Preprocess for quantum circuit
    X_processed, y_processed = preprocess_for_quantum(
        X, y, n_qubits=n_qubits, image_size=image_size, normalization=normalization,
        encoding_type=encoding_type
    )
    
    print(f"Preprocessed for quantum: {X_processed.shape}, labels: {np.unique(y_processed)}")
    
    return X_processed, y_processed


def validate_dataset(X: np.ndarray, y: np.ndarray, n_qubits: int) -> bool:
    """
    Validate dataset compatibility with quantum circuit.
    
    Args:
        X: Feature array
        y: Label array
        n_qubits: Number of qubits
    
    Returns:
        True if valid
    
    Raises:
        ValueError if validation fails
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
    
    if X.shape[1] != n_qubits:
        raise ValueError(f"Feature count {X.shape[1]} doesn't match n_qubits {n_qubits}")
    
    if not np.all((X >= 0) & (X <= 2 * np.pi + 0.1)):
        raise ValueError(f"Features must be in [0, 2π] range. Got [{X.min()}, {X.max()}]")
    
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, np.array([-1, 1])):
        raise ValueError(f"Labels must be {{-1, +1}}. Got {unique_labels}")
    
    print("Dataset validation passed")
    return True
