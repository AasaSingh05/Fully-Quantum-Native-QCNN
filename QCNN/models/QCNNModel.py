import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from QCNN.config import QuantumNativeConfig
from QCNN.layers import QuantumNativeConvolution
from QCNN.encoding import PureQuantumEncoder
from QCNN.layers import QuantumNativePooling  # Import pooling from layers

class PureQuantumNativeCNN:

    #constructor
    def __init__(self, config: QuantumNativeConfig):
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
        """Initialize all quantum circuit parameters"""
        pnp.random.seed(42)

        params = {}

        #convolutional layer
        conv_windows = QuantumNativeConvolution.get_conv_windows(self.config.image_size)
        n_windows = len(conv_windows)

        # Fix: pass the length of a window to get_kernel_param_count (parameters per kernel)
        window_size = len(conv_windows[0])
        kernel_params = QuantumNativeConvolution.get_kernel_param_count(window_size)

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
        # Fix: use window size here too
        window_size = len(conv_windows[0])
        kernel_params = QuantumNativeConvolution.get_kernel_param_count(window_size)
        kernel_size = n_windows * kernel_params

        for layer in range(self.config.n_conv_layers):
            size = kernel_size
            params[f'quantum_conv_{layer}'] = flat_params[idx:idx + size].reshape(
                (n_windows, kernel_params)
            )
            idx += size

        params['quantum_pooling'] = flat_params[idx:idx + 20]
        idx += 20

        params['quantum_classifier'] = flat_params[idx:idx + 8]
        idx += 8

        return params

    def _pure_quantum_forward(self, x: np.ndarray, params: dict) -> float:
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
            # Apply quantum convolution to all windows
            if current_image_size >= 2:
                # CHANGED: Build windows on the current active_qubits, not on global indices.
                # We first get a window pattern for the current image size (relative indices),
                # then map those relative indices onto the physical wire labels in active_qubits.
                base_windows = QuantumNativeConvolution.get_conv_windows(current_image_size)  # relative windows  # CHANGED
                conv_params = params[f'quantum_conv_{layer}']

                # Safety: if parameter tensor was sized from the initial layout, ensure indexing stays in range
                # by taking min between available conv windows and learned parameter rows.
                n_windows_runtime = min(len(base_windows), conv_params.shape[0])  # CHANGED

                # Map each relative window to the actual active wires
                # Assumes that max index in any base window < len(active_qubits)
                # If pooling reduced active_qubits, this mapping guarantees we only touch surviving wires.
                for window_idx in range(n_windows_runtime):
                    rel_window = base_windows[window_idx]
                    # Validate window fits in current active set
                    if len(rel_window) == 0:
                        continue
                    if max(rel_window) >= len(active_qubits):
                        # Skip any window that would exceed current active wires
                        continue
                    # Map relative index to physical wire ID
                    window_qubits = [active_qubits[i] for i in rel_window]  # CHANGED
                    QuantumNativeConvolution.quantum_conv2d_kernel(
                        conv_params[window_idx], window_qubits
                    )

            # Quantum pooling between layers (except last)
            if layer < self.config.n_conv_layers - 1:
                n_qubits_current = len(active_qubits)
                n_keep = max(1, int(n_qubits_current * (1 - pooling_reduction)))

                # Select disjoint qubit sets for pooling inputs and outputs
                input_qubits = active_qubits[:n_keep]
                output_qubits = active_qubits[n_keep: n_keep * 2]

                # CHANGED: If not enough qubits to form output pairs, fall back to keeping the first n_keep.
                if len(output_qubits) < n_keep:
                    # Pair as many as possible; if odd, keep available outputs.
                    output_qubits = active_qubits[n_keep:n_qubits_current]

                QuantumNativePooling.quantum_unitary_pooling(
                    params['quantum_pooling'], input_qubits=input_qubits, output_qubits=output_qubits
                )

                # CHANGED: Only the output_qubits survive; use them as inputs for the next layer.
                active_qubits = list(output_qubits)  # CHANGED

                # CHANGED: Update the logical image size to match the reduced active set length if needed.
                # Ensure current_image_size is not larger than the available relative indexing space.
                current_image_size = max(
                    2,
                    min(int(current_image_size * (1 - pooling_reduction)), len(active_qubits))
                )  # CHANGED

        # Step 3: Final quantum classification
        classifier_params = params['quantum_classifier']

        for i, param in enumerate(classifier_params[:len(active_qubits)]):
            qml.RY(param, wires=active_qubits[i])

        if len(active_qubits) >= 2:
            qml.CNOT(wires=[active_qubits[0], active_qubits[1]])

        # Step 4: Pure quantum measurement
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

        quantum_predictions = pnp.array(quantum_predictions)
        y_batch = pnp.array(y_batch)

        quantum_loss = pnp.mean((quantum_predictions - y_batch) ** 2)

        quantum_penalty = 0
        for param_set in self.quantum_params.values():
            quantum_penalty += pnp.sum(param_set ** 2)

        return quantum_loss + 0.001 * quantum_penalty
        # return quantum_loss
