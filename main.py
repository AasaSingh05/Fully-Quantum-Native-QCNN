#!/usr/bin/env python3
# Main execution script for Pure Quantum Native QCNN

import sys
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cProfile
import pstats
import numpy as np, random

# setting randomness seeds for reproducibility
np.random.seed(42)
random.seed(42)

class Logger(object):
    """Duplicates stdout/stderr to a file."""
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if hasattr(self, 'log'):
            self.log.close()

try:
    import pennylane as qml
    qml.set_seed(42)
except:
    pass

# Add the QCNN package directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import all modules
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from QCNN.utils.dataset_generator import generate_quantum_binary_dataset
from QCNN.utils.dataset_loader import load_dataset
from QCNN.utils.metadata_logger import save_metadata

    # Note: No top-level prints here to avoid clutter in multiprocess workers


def main(train_sample_size=None, use_bce=True, dataset_path=None, dataset_type='synthetic', 
         encoding='feature_map', image_size=None, log_file="training_log.txt", summary_log_file="training_summary.txt",
         learning_rate=None):
    """Main execution function
    
    Args:
        train_sample_size (int or None): Number of training samples to use.
        use_bce (bool): If True, trainer optimizes BCE; else MSE.
        dataset_path (str or None): Path to custom dataset file
        dataset_type (str): Type of dataset ('synthetic', 'npz', 'csv', 'mnist')
        encoding (str): Encoding type ('auto', 'feature_map', 'amplitude', 'patch')
        image_size (int or None): Width/height of square image
        log_file (str): Path to output log file
        summary_log_file (str): Path to summary output log file
    """
    # Initialize logger to capture all output to file
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print("Entered main() function.")

    # Step 1: Configuration
    print("\nStep 1: Initializing Quantum Configuration...")
    # Use factory to auto-configure based on image size and desired encoding
    if image_size is None:
        image_size = 4 if dataset_type == 'synthetic' else 28 # Default for MNIST/others
    
    config = QuantumNativeConfig.from_image_size(image_size, encoding)
    if learning_rate is not None:
        config.learning_rate = learning_rate
    
    print(f" Configuration: {config.n_qubits} qubits, {config.n_conv_layers} quantum layers")
    print(f" Encoding: {config.encoding_type}")
    print(f" Image Size: {config.image_size}x{config.image_size}")
    print(f" Training: {config.n_epochs} epochs, lr={config.learning_rate}")

    # save metadata after config is created
    save_metadata("Results/metadata.json", config)

    # Step 2: Load dataset (synthetic or custom)
    os.makedirs("Results", exist_ok=True)
    
    if dataset_type == 'synthetic':
        # Use synthetic dataset (backward compatibility)
        clean_dataset_path = os.path.join("Results", "quantum_dataset_clean_seed42.npz")
        force_regenerate = False
        
        if force_regenerate or not os.path.exists(clean_dataset_path):
            print("\nGenerating Quantum Binary Dataset (seed=42)...")
            np.random.seed(42)
            random.seed(42)
            
            X_quantum, y_quantum = generate_quantum_binary_dataset(
                n_samples=400,
                image_size=config.image_size
            )
            np.savez(clean_dataset_path, X=X_quantum, y=y_quantum)
            print(f"Saved dataset to '{clean_dataset_path}'")
        else:
            print(f"\nLoading synthetic dataset from '{clean_dataset_path}'...")
            data = np.load(clean_dataset_path)
            X_quantum, y_quantum = data['X'], data['y']
    
    else:
        # Load custom dataset
        print(f"\nStep 2: Loading Data ({dataset_type})...")
        X_quantum, y_quantum = load_dataset(
            source=dataset_path,
            dataset_type=dataset_type,
            n_qubits=config.n_qubits,
            image_size=config.image_size,
            normalization=config.preprocessing_mode,
            encoding_type=config.encoding_type,
            classes=tuple(args.classes)
        )
    
    # NEW: Limit total dataset size if requested, BEFORE splitting and expensive preprocessing.
    # This ensuring both train and test sets are small enough for quantum simulation.
    if train_sample_size is not None and len(X_quantum) > train_sample_size:
        # We take a slightly larger pool so that after 70/30 split, 
        # the training set is exactly train_sample_size.
        total_needed = int(train_sample_size / 0.7)
        if total_needed < len(X_quantum):
            print(f"  Limiting total dataset to {total_needed} samples for efficiency (target train: {train_sample_size})...")
            indices = np.random.choice(len(X_quantum), total_needed, replace=False)
            X_quantum = X_quantum[indices]
            y_quantum = y_quantum[indices]

    # Ensure all data (including synthetic) is preprocessed for the model if not already
    from QCNN.utils.data_preprocessing import preprocess_for_quantum
    if dataset_type == 'synthetic':
        print(f"  Preprocessing synthetic dataset for {config.encoding_type}...")
        X_quantum, y_quantum = preprocess_for_quantum(
            X_quantum, y_quantum, 
            n_qubits=config.n_qubits, 
            image_size=config.image_size,
            normalization=config.preprocessing_mode,
            encoding_type=config.encoding_type
        )
    
    print("Dataset ready.")
    # Auto-downsample only when not using amplitude encoding,
    # where we want to preserve as much image information as
    # possible for the quantum state.
    if (
        X_quantum.ndim == 3
        and X_quantum.shape[1] > 16
        and config.encoding_type not in ('amplitude', 'patch')
    ):
        print(f"  Detected high resolution ({X_quantum.shape[1:]}). Downsampling for quantum efficiency...")
        # Simple 2x2 mean pooling downsampling
        h, w = X_quantum.shape[1], X_quantum.shape[2]
        X_quantum = X_quantum[:, :h//2*2, :w//2*2].reshape(-1, h//2, 2, w//2, 2).mean(axis=(2, 4))
        config.image_size = h // 2
        print(f"  New resolution: {X_quantum.shape[1:]}")
    print(f"Shape: {X_quantum.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y_quantum, return_counts=True)))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y_quantum, test_size=0.3, random_state=42, stratify=y_quantum
    )

    # Limit training samples if train_sample_size parameter set
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train = X_train[:train_sample_size]
        y_train = y_train[:train_sample_size]

    print(f" Training samples used: {len(X_train)}")
    print(f" Test samples: {len(X_test)}")
    print(f" Class distribution (training): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Step 3: Model initialization
    print("\n Step 3: Initializing Pure Quantum CNN...")
    quantum_model = PureQuantumNativeCNN(config)

    total_params = sum(np.prod(p.shape) for p in quantum_model.quantum_params.values())
    print(f" Quantum parameters: {total_params}")
    print(f" Hilbert space: 2^{config.n_qubits} = {2 ** config.n_qubits} dimensions")

    # Step 4: Training
    print("\n Step 4: Training Pure Quantum CNN...")
    trainer = QuantumNativeTrainer(learning_rate=config.learning_rate, use_bce=use_bce)
    trained_model = trainer.train_pure_quantum_cnn(
        quantum_model, X_train, y_train, X_test, y_test, log_filepath=log_file, summary_filepath=summary_log_file
    )

    # Step 5: Evaluation
    print("\n Step 5: Evaluating Quantum Model...")

    # Use the exact test representation that was used during training, if available.
    X_eval = X_test
    y_eval = y_test
    if hasattr(trained_model, "_quantum_preprocessed_test"):
        try:
            cached_X, cached_y = trained_model._quantum_preprocessed_test, y_test
            # In case future extensions also cache labels, handle tuple form.
            if isinstance(trained_model._quantum_preprocessed_test, tuple):
                cached_X, cached_y = trained_model._quantum_preprocessed_test
            if len(cached_X) == len(y_test):
                X_eval = cached_X
                y_eval = cached_y
        except Exception:
            # Fallback gracefully to original test set on any mismatch
            X_eval = X_test
            y_eval = y_test

    predictions = trained_model.quantum_predict_batch(X_eval)
    accuracy = np.mean(predictions == y_eval)
    print(f" Final Quantum Accuracy: {accuracy:.1%}")

    # Confusion matrix
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == -1) & (y_test == -1))
    fp = np.sum((predictions == 1) & (y_test == -1))
    fn = np.sum((predictions == -1) & (y_test == 1))

    print(f"\n Quantum Confusion Matrix:")
    print(f"   True Positives: {tp}")
    print(f"   True Negatives: {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")

    # Plot training history and save graphs
    try:
        graphs_dir = os.path.join('Results', 'Graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        graph_path = os.path.join(graphs_dir, 'quantum_training_results.png')

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(trained_model.training_history['loss'])
        plt.title('Pure Quantum Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Quantum Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(trained_model.training_history['accuracy'])
        plt.title('Pure Quantum Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Quantum Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"\n Training plots saved as '{graph_path}'")
    except Exception as e:
        print(f"\n  Plotting failed: {e}")

    print("\n" + "-" * 60)
    print("EXECUTION OF QCNN COMPLETE!")
    print("-" * 60)
    print(f" Achieved {accuracy:.1%} accuracy with 100% quantum operations")
    return trained_model, accuracy


if __name__ == "__main__":
    print("Starting main execution script...")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Quantum CNN with custom or synthetic datasets')
    parser.add_argument('--dataset', type=str, default='images',
                       choices=['auto', 'synthetic', 'npz', 'csv', 'mnist', 'images', 'idx'],
                       help='Type of dataset to use')
    parser.add_argument('--path', type=str, default=None,
                       help='Path to custom dataset file or directory')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of training samples to use')
    parser.add_argument('--use-mse', action='store_true',
                       help='Use Mean Squared Error loss instead of BCE')
    parser.add_argument('--encoding', type=str, default='auto',
                       choices=['auto', 'feature_map', 'amplitude', 'patch'],
                       help='Encoding strategy to use')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Width/height of square input images')
    parser.add_argument('--classes', type=int, nargs=2, default=[0, 1],
                       help='Two classes to use for binary classification (default: 0 1)')
    parser.add_argument('--log-file', type=str, default='training_log.txt',
                       help='File to save training logs (default: training_log.txt)')
    parser.add_argument('--summary-log', type=str, default='training_summary.txt',
                       help='File to save epoch summaries (default: training_summary.txt)')
    parser.add_argument('--no-profile', action='store_true',
                       help='Disable cProfile performance profiling')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for quantum optimization')
    
    args = parser.parse_args()
    
    try:
        if not args.no_profile:
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Run main with arguments
        model, acc = main(
            train_sample_size=args.samples,
            use_bce=not args.use_mse,
            dataset_path=args.path,
            dataset_type=args.dataset,
            encoding=args.encoding,
            image_size=args.image_size,
            log_file=args.log_file,
            summary_log_file=args.summary_log,
            learning_rate=args.learning_rate
        )
        
        if not args.no_profile:
            profiler.disable()
            print(f"\nExecution completed successfully! Accuracy: {acc:.1%}")
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats(20)
        else:
            print(f"\nExecution completed successfully! Accuracy: {acc:.1%}")
    
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()