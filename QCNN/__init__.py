"""
Pure Quantum Native QCNN Binary Classifier
100% quantum operations - no classical neural network components
"""

__version__ = "1.0.0"
__author__ = "Quantum Research Team"

# Main imports for easy access
from .config.Qconfig import QuantumNativeConfig
from .models.QCNNModel import PureQuantumNativeCNN
from .training.Qtrainer import QuantumNativeTrainer
from .utils.dataset_generator import generate_quantum_binary_dataset

__all__ = [
    "QuantumNativeConfig",
    "PureQuantumNativeCNN", 
    "QuantumNativeTrainer",
    "generate_quantum_binary_dataset"
]

def quick_start():
    """Quick start example"""
    config = QuantumNativeConfig()
    model = PureQuantumNativeCNN(config)
    return model, config
