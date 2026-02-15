import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from QCNN.config import QuantumNativeConfig
from QCNN.layers import QuantumNativeConvolution
from QCNN.encoding import PureQuantumEncoder
from QCNN.layers import QuantumNativePooling  # Import pooling from layers


class PureQuantumNativeCNN:
    """
    Fully quantum-native convolutional neural network.
    Implements encoding, convolution, pooling, and classification
    entirely using unitary quantum operations with shared kernels
    and differentiable PennyLane QNodes.
    """

    #constructor
    def __init__(self, config: QuantumNativeConfig):
        """
        Initialize QCNN with device, parameters, and differentiable QNode.

        Args:
            config: QuantumNativeConfig containing qubit count, image size,
                    number of layers, encoding type, and optimizer settings.
        """
        self.config = config
        self.device = qml.device(config.device, wires=config.n_qubits)

        # Store number of qubits for parameter calculation
        self.num_qubits = config.n_qubits

        # Initialize pure quantum parameters
        self.quantum_params = self._initialize_quantum_parameters()

        # Create pure quantum circuit with explicit QNode decorator for differentiation
        @qml.qnode(self.device, interface='autograd')  # interface='jax')
        def quantum_circuit(x, flat_params):
            # Unflatten parameters from flat vector
            params = self._unflatten_params(flat_params)
            return self._pure_quantum_forward(x, params)

        self.quantum_circuit = quantum_circuit

        # Training metrics
        self.training_history = {'loss': [], 'accuracy': []}

    def _initialize_quantum_parameters(self) -> dict[str, pnp.ndarray]:
        """
        Allocate and initialize all trainable quantum parameters:
        - Shared 2×2 convolution kernels per layer
        - Unitary pooling parameters
        - Final shallow classifier head

        Returns:
            dict mapping parameter names to PennyLane arrays.
        """
        pnp.random.seed(42)

        params = {}

        # Convolutional layer: SHARED 2x2 kernel per layer.
        # Each kernel acts on 4 qubits with shape (4, depth, 2) => [RY, RZ] angles per depth.
        # Keep depth small for stability.
        conv_depth = 1
        kernel_shape = (4, conv_depth, 2)
        for layer in range(self.config.n_conv_layers):
            kernel_tensor = pnp.array(np.random.normal(0, 0.1, kernel_shape), requires_grad=True)
            params[f'quantum_conv_kernel_{layer}'] = kernel_tensor

        # Pooling layer parameters:
        # We consume 3 angles per (keep, discard) pair in QuantumNativePooling.quantum_unitary_pooling.
        max_pairs = self.num_qubits // 2
        pool_angles = 3 * max_pairs
        params['quantum_pooling'] = pnp.array(np.random.normal(0, 0.1, pool_angles), requires_grad=True)

        # Fully connected classifier layer (on up to last 2 active qubits)
        params['quantum_classifier'] = pnp.array(np.random.normal(0, 0.1, 8), requires_grad=True)

        return params

    def _flatten_params(self, params: dict[str, pnp.ndarray]) -> pnp.ndarray:
        """
        Flatten dictionary of parameter arrays into a single vector.

        Args:
            params: dict mapping parameter names to tensors.

        Returns:
            1D PennyLane array containing all parameters in order.
        """
        return pnp.concatenate([p.flatten() for p in params.values()])

    def _unflatten_params(self, flat_params: pnp.ndarray) -> dict[str, pnp.ndarray]:
        """
        Convert flat parameter vector back into structured dictionary.

        Args:
            flat_params: 1D flattened parameter vector.

        Returns:
            dict mapping names to reshaped parameter tensors.
        """
        params = {}
        idx = 0

        # Convs: each layer kernel is (4, depth, 2)
        conv_depth = 1
        kernel_size = 4 * conv_depth * 2
        for layer in range(self.config.n_conv_layers):
            size = kernel_size
            slice_flat = flat_params[idx:idx + size]
            params[f'quantum_conv_kernel_{layer}'] = slice_flat.reshape(4, conv_depth, 2)
            idx += size

        # Pooling params: fixed length allocated above
        max_pairs = self.num_qubits // 2
        pool_angles = 3 * max_pairs
        params['quantum_pooling'] = flat_params[idx:idx + pool_angles]
        idx += pool_angles

        # Classifier: fixed length 8
        params['quantum_classifier'] = flat_params[idx:idx + 8]
        idx += 8

        return params

    def _pure_quantum_forward(self, x: np.ndarray, params: dict) -> float:
        """
        Full QCNN forward pass using only quantum operations.
        Applies encoding, convolution, pooling, and shallow classification
        before returning the expectation value ⟨Z⟩ on the readout qubit.

        Args:
            x: input classical 4×4 data sample
            params: structured parameter dictionary for all QCNN layers

        Returns:
            float expectation value for classification.
        """
        all_qubits = list(range(self.config.n_qubits))

        # Step 1: Pure quantum data encoding
        if self.config.encoding_type == 'amplitude':
            PureQuantumEncoder.amplitude_encoding(x, all_qubits)
        else:
            PureQuantumEncoder.quantum_feature_map(x, all_qubits)

        # Step 2: Quantum convolutional layers and pooling
        active_qubits = all_qubits.copy()
        current_image_size = self.config.image_size

        pooling_reduction = getattr(self.config, 'pooling_reduction', 0.5)

        for layer in range(self.config.n_conv_layers):
            # Apply quantum convolution to all windows on the CURRENT active layout
            if current_image_size >= 2 and len(active_qubits) >= 4:
                base_windows = QuantumNativeConvolution.get_conv_windows(current_image_size)  # relative windows
                kernel = params[f'quantum_conv_kernel_{layer}']  # shared weights: (4, depth, 2)

                for rel_window in base_windows:
                    # Skip any window beyond current active wires
                    if max(rel_window) >= len(active_qubits):
                        continue
                    window_qubits = [active_qubits[i] for i in rel_window]
                    QuantumNativeConvolution.quantum_conv2d_kernel(kernel, window_qubits)

            # Quantum pooling between layers (except last)
            if layer < self.config.n_conv_layers - 1:
                n_qubits_current = len(active_qubits)
                if n_qubits_current < 2:
                    break

                # Default pairing: consecutive pairs
                pairs = QuantumNativePooling.make_pairing(active_qubits)
                if len(pairs) == 0:
                    break

                keep = [k for (k, _) in pairs]
                discard = [d for (_, d) in pairs]

                QuantumNativePooling.quantum_unitary_pooling(
                    params['quantum_pooling'],
                    input_qubits=keep,
                    output_qubits=discard
                )

                # Only the keep wires survive for the next stage
                active_qubits = keep

                # Update logical image size after pooling
                if current_image_size > 2:
                    current_image_size = max(2, current_image_size // 2)

        # Step 3: Final quantum classification
        classifier_params = params['quantum_classifier']

        # Apply RY on up to the first two active qubits
        for i, q in enumerate(active_qubits[:2]):
            angle = classifier_params[i % len(classifier_params)]
            qml.RY(angle, wires=q)

        # Light entanglement if possible
        if len(active_qubits) >= 2:
            qml.CNOT(wires=[active_qubits[0], active_qubits[1]])

        # Final tweak on readout
        if len(active_qubits) >= 1:
            qml.RZ(classifier_params[-1], wires=active_qubits[0])

        # Measurement
        return qml.expval(qml.PauliZ(active_qubits[0]))

    def quantum_predict_single(self, x: np.ndarray) -> float:
        """
        Predict ⟨Z⟩ for a single sample using the pure quantum circuit.

        Args:
            x: input 4×4 sample

        Returns:
            float: expectation value of the readout qubit.
        """
        flat_params = self._flatten_params(self.quantum_params)
        return self.quantum_circuit(x, flat_params)

    def quantum_predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch quantum prediction with {-1,1} output encoding.

        Args:
            X: array of samples

        Returns:
            numpy array of binary predictions.
        """
        predictions = []
        for x in X:
            quantum_output = self.quantum_predict_single(x)
            binary_pred = 1 if quantum_output > 0 else -1
            predictions.append(binary_pred)
        return np.array(predictions)

    def quantum_loss_function(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Computes pure quantum MSE loss + small L2 penalty.
        Evaluated entirely from quantum predictions.

        Args:
            X_batch: minibatch of samples
            y_batch: target labels in {-1,1}

        Returns:
            scalar loss value
        """
        quantum_predictions = []
        for x in X_batch:
            quantum_output = self.quantum_predict_single(x)
            quantum_predictions.append(quantum_output)

        quantum_predictions = pnp.array(quantum_predictions)
        y_batch = pnp.array(y_batch)

        quantum_loss = pnp.mean((quantum_predictions - y_batch) ** 2)

        quantum_penalty = 0
        for param_set in self.quantum_params.values():
            quantum_penalty += pnp.sum(param_set ** 2)

        return quantum_loss + 0.001 * quantum_penalty
        # return quantum_loss
