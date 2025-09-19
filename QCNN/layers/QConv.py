import numpy as np
import pennylane as qml


class QuantumNativeConvolution:
    @staticmethod
    def quantum_conv2d_kernel(params: np.ndarray, qubits: list[int]) -> None:
        """
        Quantum convolution kernel for any number of qubits (multiple of 4).
        Applies quantum rotations and nearest-neighbor entanglement.

        Args:
            params (np.ndarray): Array of parameters, length = 4 * len(qubits)
            qubits (list[int]): List of qubit indices (must be multiple of 4)
        """
        n = len(qubits)
        if n % 4 != 0:
            raise ValueError("Number of qubits must be a multiple of 4")

        # Number of parameter sets = number of qubits
        # We will share parameters across qubits in groups of 2 to reduce parameters
        params = np.array(params)
        if len(params) != 4 * n:
            raise ValueError(f"Expected {4 * n} parameters, got {len(params)}")

        # Each block of 4 qubits (like a 2x2 kernel) will share similar structure
        # Reshape parameters for readability: (layers=4, qubits)
        param_layers = params.reshape(4, n)

        # Layer 1 & 3 rotations combined with parameter sharing every 2 qubits
        for i, qubit in enumerate(qubits):
            # Layer 1 rotations (RY, RZ) sharing parameter every 2 qubits
            qml.RY(param_layers[0, i], wires=qubit)
            qml.RZ(param_layers[1, i], wires=qubit)
            # Layer 3 rotation (RX)
            qml.RX(param_layers[2, i], wires=qubit)

        # Nearest neighbor entanglement (edges of qubit grid)
        # Assuming qubits arranged in 2D grid with width = sqrt(n)
        width = int(np.sqrt(n))
        if width * width != n:
            raise ValueError("Qubits should form a perfect square grid for entanglement")

        # Horizontal edges
        for row in range(width):
            for col in range(width - 1):
                qml.CNOT(wires=[qubits[row * width + col], qubits[row * width + col + 1]])

        # Vertical edges
        for col in range(width):
            for row in range(width - 1):
                qml.CNOT(wires=[qubits[row * width + col], qubits[(row + 1) * width + col]])

        # Diagonal entanglement
        for row in range(width - 1):
            for col in range(width - 1):
                qml.CNOT(wires=[qubits[row * width + col], qubits[(row + 1) * width + col + 1]])
                qml.CNOT(wires=[qubits[row * width + col + 1], qubits[(row + 1) * width + col]])

        # Layer 4 final rotations (RY)
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
