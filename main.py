#!/usr/bin/env python3
#Main execution script for Pure Quantum Native QCNN

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cProfile
import pstats

# Add the QCNN package directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import all modules
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from QCNN.utils.dataset_generator import generate_quantum_binary_dataset

print("Starting main execution script...")  # Debug to monitor re-imports

def main(train_sample_size=None):
    """Main execution function
    
    Args:
        train_sample_size (int or None): Number of training samples to use. If None,
                                         all training data is used.
    """
    print("Entered main() function.")  # Debug print

    # Step 1: Configuration
    print("\n📋 Step 1: Initializing Quantum Configuration...")
    config = QuantumNativeConfig()
    print(f"✅ Configuration: {config.n_qubits} qubits, {config.n_conv_layers} quantum layers")
    print(f"✅ Encoding: {config.encoding_type}")
    print(f"✅ Training: {config.n_epochs} epochs, lr={config.learning_rate}")

    # Define dataset save path
    dataset_path = os.path.join('Results', 'quantum_dataset.npz')
    os.makedirs('Results', exist_ok=True)

    # Step 2: Generate or Load quantum dataset
    if os.path.exists(dataset_path):
        print(f"\n📥 Loading quantum dataset from '{dataset_path}'...")
        data = np.load(dataset_path)
        X_quantum, y_quantum = data['X'], data['y']
    else:
        print("\n📊 Step 2: Generating Quantum Binary Dataset...")
        X_quantum, y_quantum = generate_quantum_binary_dataset(
            n_samples=300, image_size=config.image_size
        )
        np.savez(dataset_path, X=X_quantum, y=y_quantum)
        print(f"✅ Saved quantum dataset to '{dataset_path}'")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y_quantum, test_size=0.3, random_state=42, stratify=y_quantum
    )

    # Limit training samples if train_sample_size parameter set
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train = X_train[:train_sample_size]
        y_train = y_train[:train_sample_size]

    print(f"✅ Training samples used: {len(X_train)}")
    print(f"✅ Test samples: {len(X_test)}")
    print(f"✅ Class distribution (training): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Step 3: Model initialization
    print("\n⚡ Step 3: Initializing Pure Quantum CNN...")
    quantum_model = PureQuantumNativeCNN(config)

    total_params = sum(np.prod(p.shape) for p in quantum_model.quantum_params.values())
    print(f"✅ Quantum parameters: {total_params}")
    print(f"✅ Hilbert space: 2^{config.n_qubits} = {2 ** config.n_qubits} dimensions")

    # Step 4: Training
    print("\n🚀 Step 4: Training Pure Quantum CNN...")
    trainer = QuantumNativeTrainer(learning_rate=config.learning_rate)
    trained_model = trainer.train_pure_quantum_cnn(
        quantum_model, X_train, y_train, X_test, y_test
    )

    # Step 5: Evaluation
    print("\n📈 Step 5: Evaluating Quantum Model...")
    predictions = trained_model.quantum_predict_batch(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"🎯 Final Quantum Accuracy: {accuracy:.1%}")

    # Confusion matrix
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == -1) & (y_test == -1))
    fp = np.sum((predictions == 1) & (y_test == -1))
    fn = np.sum((predictions == -1) & (y_test == 1))

    print(f"\n📊 Quantum Confusion Matrix:")
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
        print(f"\n📊 Training plots saved as '{graph_path}'")
    except Exception as e:
        print(f"\n⚠️  Plotting failed: {e}")

    print("\n" + "=" * 60)
    print("🏆 PURE QUANTUM NATIVE QCNN COMPLETE!")
    print("=" * 60)
    print(f"🎯 Achieved {accuracy:.1%} accuracy with 100% quantum operations")
    print("🔬 Quantum advantages utilized:")
    print(f"   • Exponential state space: 2^{config.n_qubits} dimensions")
    print("   • Quantum entanglement: Non-local correlations")
    print("   • Quantum superposition: Parallel computation")
    print("   • Quantum interference: Amplitude cancellation")
    print("✅ Ready for quantum hardware deployment!\n")
    print("Exiting main() function.")  # Debug print
    return trained_model, accuracy

if __name__ == "__main__":
    try:
        profiler = cProfile.Profile()
        profiler.enable()

        # You can specify train_sample_size or None to use all
        model, acc = main(train_sample_size=50)

        profiler.disable()
        print(f"\n✅ Execution completed successfully! Accuracy: {acc:.1%}")

        # Print profiling stats sorted by cumulative time
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)  # Show top 20 functions by cumulative time

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
