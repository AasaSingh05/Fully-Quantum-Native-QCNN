#!/usr/bin/env python3
"""
Example: Loading Custom Datasets for QCNN Training

This script demonstrates how to load various dataset formats and train
the Quantum CNN with your own data.
"""

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from QCNN.utils import load_dataset


def example_1_load_npz():
    """Example 1: Load dataset from NPZ file"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Loading from NPZ file")
    print("="*60)
    
    # Path to your NPZ file (must contain 'X' and 'y' arrays)
    npz_path = "Results/quantum_dataset_clean_seed42.npz"
    
    if not os.path.exists(npz_path):
        print(f"NPZ file not found: {npz_path}")
        print("Skipping this example...")
        return None
    
    config = QuantumNativeConfig()
    
    # Load and preprocess dataset
    X, y = load_dataset(
        source=npz_path,
        dataset_type='npz',
        n_qubits=config.n_qubits,
        image_size=config.image_size,
        normalization='minmax'
    )
    
    print(f"Loaded {len(X)} samples from NPZ file")
    print(f"  Shape: {X.shape}, Labels: {np.unique(y)}")
    
    return X, y, config


def example_2_load_csv():
    """Example 2: Load dataset from CSV file"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Loading from CSV file")
    print("="*60)
    
    # Create a sample CSV for demonstration
    csv_path = "examples/sample_dataset.csv"
    
    # Generate sample data
    print("Creating sample CSV file...")
    n_samples = 200
    n_features = 16  # Must match n_qubits
    
    X_sample = np.random.randn(n_samples, n_features)
    y_sample = np.random.choice([0, 1], n_samples)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(X_sample, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = y_sample
    df.to_csv(csv_path, index=False)
    
    config = QuantumNativeConfig()
    
    # Load from CSV
    X, y = load_dataset(
        source=csv_path,
        dataset_type='csv',
        n_qubits=config.n_qubits,
        image_size=config.image_size,
        normalization='minmax',
        label_column='label'  # Specify label column name
    )
    
    print(f"Loaded {len(X)} samples from CSV file")
    print(f"  Shape: {X.shape}, Labels: {np.unique(y)}")
    
    return X, y, config


def example_3_load_mnist():
    """Example 3: Load MNIST subset for binary classification"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Loading MNIST subset (digits 0 vs 1)")
    print("="*60)
    
    config = QuantumNativeConfig()
    
    try:
        # Load MNIST subset (this will download if needed)
        X, y = load_dataset(
            source='mnist',  # Special keyword for MNIST
            dataset_type='mnist',
            n_qubits=config.n_qubits,
            image_size=config.image_size,
            normalization='minmax',
            n_samples=500,  # Limit samples for quick training
            classes=(0, 1)  # Binary classification: 0 vs 1
        )
        
        print(f"Loaded {len(X)} MNIST samples")
        print(f"  Shape: {X.shape}, Labels: {np.unique(y)}")
        
        return X, y, config
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Install scikit-learn to use MNIST: pip install scikit-learn")
        return None


def example_4_custom_arrays():
    """Example 4: Use custom numpy arrays directly"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Using custom numpy arrays")
    print("="*60)
    
    config = QuantumNativeConfig()
    
    # Create your own data
    n_samples = 300
    n_features = config.n_qubits  # Must match
    
    # Example: Create synthetic data with patterns
    X_custom = np.random.randn(n_samples, n_features)
    y_custom = (X_custom[:, 0] > 0).astype(int)  # Simple rule
    
    # Load using tuple format
    X, y = load_dataset(
        source=(X_custom, y_custom),  # Pass as tuple
        dataset_type='array',
        n_qubits=config.n_qubits,
        image_size=config.image_size,
        normalization='standard'
    )
    
    print(f"Loaded {len(X)} custom samples")
    print(f"  Shape: {X.shape}, Labels: {np.unique(y)}")
    
    return X, y, config


def train_with_custom_data(X, y, config):
    """Train QCNN with custom dataset"""
    print("\n" + "="*60)
    print("TRAINING QCNN WITH CUSTOM DATA")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize model
    model = PureQuantumNativeCNN(config)
    
    # Initialize trainer
    trainer = QuantumNativeTrainer(learning_rate=config.learning_rate, use_bce=False)
    
    # Train (reduce epochs for quick demo)
    config.n_epochs = 5  # Quick demo
    trained_model = trainer.train_pure_quantum_cnn(
        model, X_train, y_train, X_test, y_test,
        log_filepath='examples/custom_training_log.txt'
    )
    
    # Evaluate
    predictions = trained_model.quantum_predict_batch(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Final Test Accuracy: {accuracy:.1%}")
    
    return trained_model


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUSTOM DATASET LOADING EXAMPLES FOR QCNN")
    print("="*70)
    
    # Run examples
    examples = [
        ("NPZ File", example_1_load_npz),
        ("CSV File", example_2_load_csv),
        ("MNIST Subset", example_3_load_mnist),
        ("Custom Arrays", example_4_custom_arrays),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    # Run example 4 (custom arrays) as default
    print("\n" + "-"*70)
    print("Running Example 4: Custom Arrays (fastest)")
    print("-"*70)
    
    result = example_4_custom_arrays()
    
    if result is not None:
        X, y, config = result
        
        # Optionally train
        print("\nWould you like to train with this data? (This is a demo, skipping training)")
        print("To train, uncomment the line below:")
        print("# train_with_custom_data(X, y, config)")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE!")
    print("="*70)
    print("\nTo use your own dataset:")
    print("1. Prepare your data as NPZ, CSV, or numpy arrays")
    print("2. Use load_dataset() with appropriate parameters")
    print("3. Pass to trainer.train_pure_quantum_cnn()")
    print("\nSee this script for detailed examples!")
