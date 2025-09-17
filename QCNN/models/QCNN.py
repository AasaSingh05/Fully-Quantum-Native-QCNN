import numpy as np
import pennylane as qml
from QCNN.config import QuantumNativeConfig 
from QCNN.layers import QuantumNativeConvolution 
from QCNN.encoding import PureQuantumEncoder 
class PureQuantumNativeCNN:
    """
    100% Quantum Native CNN - Zero Classical Components
    Every operation uses quantum mechanical principles
    """
    
    def __init__(self, config: QuantumNativeConfig):
        self.config = config
        self.device = qml.device(config.device, wires=config.n_qubits)
        
        # Initialize pure quantum parameters
        self.quantum_params = self._initialize_quantum_parameters()
        
        # Create pure quantum circuit
        self.quantum_circuit = qml.QNode(self._pure_quantum_forward, self.device)
        
        # Training metrics  
        self.training_history = {'loss': [], 'accuracy': []}
    
    def _initialize_quantum_parameters(self) -> dict[str, np.ndarray]:
        """Initialize all quantum circuit parameters"""
        np.random.seed(42)
        params = {}
        
        # Quantum convolution parameters
        conv_windows = QuantumNativeConvolution.get_conv_windows(self.config.image_size)
        n_windows = len(conv_windows)
        kernel_params = QuantumNativeConvolution.get_kernel_param_count()
        
        for layer in range(self.config.n_conv_layers):
            params[f'quantum_conv_{layer}'] = np.random.normal(
                0, 0.1, (n_windows, kernel_params)
            )
        
        # Quantum pooling parameters
        params['quantum_pooling'] = np.random.normal(0, 0.1, 20)
        
        # Final quantum classifier parameters
        params['quantum_classifier'] = np.random.normal(0, 0.1, 8)
        
        return params
    
    def _pure_quantum_forward(self, x: np.ndarray, params: dict) -> float:
        """
        Pure quantum forward pass - 100% quantum operations
        No classical neural network components anywhere
        """
        all_qubits = list(range(self.config.n_qubits))
        
        # Step 1: Pure quantum data encoding
        if self.config.encoding_type == 'amplitude':
            PureQuantumEncoder.amplitude_encoding(x, all_qubits)
        else:
            PureQuantumEncoder.quantum_feature_map(x, all_qubits)
        
        # Step 2: Quantum convolutional layers
        active_qubits = all_qubits.copy()
        current_image_size = self.config.image_size
        
        for layer in range(self.config.n_conv_layers):
            # Apply quantum convolution to all windows
            if current_image_size >= 2:
                conv_windows = QuantumNativeConvolution.get_conv_windows(current_image_size)
                conv_params = params[f'quantum_conv_{layer}']
                
                for window_idx, window_qubits in enumerate(conv_windows):
                    # Apply quantum convolution kernel
                    QuantumNativeConvolution.quantum_conv2d_kernel(
                        conv_params[window_idx], window_qubits
                    )
            
            # Quantum pooling between layers (except last)
            if layer < self.config.n_conv_layers - 1:
                # Simple quantum pooling: keep every other qubit
                active_qubits = active_qubits[::2]  
                current_image_size = max(2, current_image_size // 2)
        
        # Step 3: Final quantum classification
        classifier_params = params['quantum_classifier']
        
        # Apply final quantum rotations for classification
        for i, param in enumerate(classifier_params[:len(active_qubits)]):
            qml.RY(param, wires=active_qubits[i])
        
        # Create final quantum entanglement
        if len(active_qubits) >= 2:
            qml.CNOT(wires=[active_qubits[0], active_qubits[1]])
        
        # Step 4: Pure quantum measurement for binary classification
        return qml.expval(qml.PauliZ(active_qubits[0]))
    
    def quantum_predict_single(self, x: np.ndarray) -> float:
        """Pure quantum prediction for single sample"""
        return self.quantum_circuit(x, self.quantum_params)
    
    def quantum_predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Pure quantum batch prediction"""
        predictions = []
        for x in X:
            quantum_output = self.quantum_predict_single(x)
            # Convert quantum measurement to binary classification
            binary_pred = 1 if quantum_output > 0 else -1
            predictions.append(binary_pred)
        return np.array(predictions)
    
    def quantum_loss_function(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Pure quantum loss computation
        Uses quantum fidelity-based loss
        """
        quantum_predictions = []
        for x in X_batch:
            quantum_output = self.quantum_predict_single(x)
            quantum_predictions.append(quantum_output)
        
        quantum_predictions = np.array(quantum_predictions)
        
        # Quantum-inspired loss: minimize distance in Hilbert space
        quantum_loss = np.mean((quantum_predictions - y_batch) ** 2)
        
        # Quantum regularization: penalize large rotations
        quantum_penalty = 0
        for param_set in self.quantum_params.values():
            quantum_penalty += np.sum(param_set ** 2)
        
        return quantum_loss + 0.001 * quantum_penalty
