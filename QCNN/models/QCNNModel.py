import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from QCNN.config import QuantumNativeConfig 
from QCNN.layers import QuantumNativeConvolution 
from QCNN.encoding import PureQuantumEncoder 

class PureQuantumNativeCNN:
    
    #constructor  
    def __init__(self, config: QuantumNativeConfig):
        self.config = config
        self.device = qml.device(config.device, wires=config.n_qubits)
        
        # Initialize pure quantum parameters
        self.quantum_params = self._initialize_quantum_parameters()
        
        # Create pure quantum circuit with explicit QNode decorator for differentiation
        @qml.qnode(self.device, diff_method="parameter-shift")
        def quantum_circuit(x, flat_params):
            # Unflatten parameters from flat vector
            params = self._unflatten_params(flat_params)
            return self._pure_quantum_forward(x, params)
        self.quantum_circuit = quantum_circuit
        
        # Training metrics  
        self.training_history = {'loss': [], 'accuracy': []}
    
    def _initialize_quantum_parameters(self) -> dict[str, pnp.ndarray]:
        """Initialize all quantum circuit parameters"""
        pnp.random.seed(42)
        params = {}

        #convolutional layer
        conv_windows = QuantumNativeConvolution.get_conv_windows(self.config.image_size)
        n_windows = len(conv_windows)
        kernel_params = QuantumNativeConvolution.get_kernel_param_count()

        for layer in range(self.config.n_conv_layers):
            param_array = pnp.array(
                np.random.normal(0, 0.1, (n_windows, kernel_params)), requires_grad=True
            )
            params[f'quantum_conv_{layer}'] = param_array

        #pooling layer
        params['quantum_pooling'] = pnp.array(np.random.normal(0, 0.1, 20), requires_grad=True)
        
        #Fully connected classifier layer
        params['quantum_classifier'] = pnp.array(np.random.normal(0, 0.1, 8), requires_grad=True)

        return params
    
    def _flatten_params(self, params: dict[str, pnp.ndarray]) -> pnp.ndarray:
        """Flatten dictionary of params into single vector"""
        return pnp.concatenate([p.flatten() for p in params.values()])
    
    def _unflatten_params(self, flat_params: pnp.ndarray) -> dict[str, pnp.ndarray]:
        """Unflatten flat vector back into dictionary of params"""
        params = {}
        idx = 0
        
        conv_windows = QuantumNativeConvolution.get_conv_windows(self.config.image_size)
        n_windows = len(conv_windows)
        kernel_params = QuantumNativeConvolution.get_kernel_param_count()
        kernel_size = n_windows * kernel_params
        
        for layer in range(self.config.n_conv_layers):
            size = kernel_size
            params[f'quantum_conv_{layer}'] = flat_params[idx:idx+size].reshape((n_windows, kernel_params))
            idx += size
        
        params['quantum_pooling'] = flat_params[idx:idx+20]
        idx += 20
        
        params['quantum_classifier'] = flat_params[idx:idx+8]
        idx += 8
        
        return params
    
    def _pure_quantum_forward(self, x: np.ndarray, params: dict) -> float:
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
        flat_params = self._flatten_params(self.quantum_params)
        return self.quantum_circuit(x, flat_params)
    
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
