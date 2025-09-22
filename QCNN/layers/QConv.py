import numpy as np
import pennylane as qml

class QuantumNativeConvolution:
    @staticmethod
    def quantum_conv2d_kernel(params: np.ndarray, qubits: list[int]) -> None:
        """
        Quantum convolution kernel for any number of qubits (multiple of 4).
        Applies parameter scaling and quantum rotations with nearest-neighbor entanglement.

        Args:
            params (np.ndarray): Array of parameters, length = 4 * len(qubits)
            qubits (list[int]): List of qubit indices (must be multiple of 4)
        """
        n = len(qubits)
        if n % 4 != 0:
            raise ValueError("Number of qubits must be a multiple of 4")

        params = np.array(params)
        if len(params) != 4 * n:
            raise ValueError(f"Expected {4 * n} parameters, got {len(params)}")

        # Scale parameters from data range to [0, 2pi]
        params_scaled = (params - np.min(params)) / (np.ptp(params) + 1e-10) * 2 * np.pi

        param_layers = params_scaled.reshape(4, n)

        # Layer 1 & 3 rotations with scaled parameters
        for i, qubit in enumerate(qubits):
            qml.RY(param_layers[0, i], wires=qubit)
            qml.RZ(param_layers[1, i], wires=qubit)
            qml.RX(param_layers[2, i], wires=qubit)

        # Compute grid width and check for perfect square
        width = int(np.sqrt(n))
        if width * width != n:
            raise ValueError("Qubits should form a perfect square grid for entanglement")

        # Horizontal CNOT entanglements
        for row in range(width):
            for col in range(width - 1):
                qml.CNOT(wires=[qubits[row * width + col], qubits[row * width + col + 1]])

        # Vertical CNOT entanglements
        for col in range(width):
            for row in range(width - 1):
                qml.CNOT(wires=[qubits[row * width + col], qubits[(row + 1) * width + col]])

        # Diagonal CNOT entanglements
        for row in range(width - 1):
            for col in range(width - 1):
                qml.CNOT(wires=[qubits[row * width + col], qubits[(row + 1) * width + col + 1]])
                qml.CNOT(wires=[qubits[row * width + col + 1], qubits[(row + 1) * width + col]])

        # Final rotations layer
        for i, qubit in enumerate(qubits):
            qml.RY(param_layers[3, i], wires=qubit)

    @staticmethod
    def get_kernel_param_count(num_qubits: int) -> int:
        """Number of parameters scales with 4 * num_qubits"""
        return 4 * num_qubits

    @staticmethod
    def get_conv_windows(image_size: int) -> list[list[int]]:
        """Generate all 2x2 windows for a square grid qubit layout"""
        windows = []
        for row in range(image_size - 1):
            for col in range(image_size - 1):
                top_left = row * image_size + col
                top_right = row * image_size + (col + 1)
                bottom_left = (row + 1) * image_size + col
                bottom_right = (row + 1) * image_size + (col + 1)
                windows.append([top_left, top_right, bottom_left, bottom_right])
        return windows
