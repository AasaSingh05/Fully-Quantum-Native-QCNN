import numpy as np
import pennylane as qml


class QuantumNativeConvolution:
    @staticmethod
    def quantum_conv2d_kernel(params: np.ndarray, qubits: list[int]) -> None:
        """
        Quantum convolution kernel for a 2x2 window (4 qubits).
        Applies parameterized rotations with local entanglement.

        Args:
            params (np.ndarray): Trainable kernel parameters for a single 2x2 window.
                                 Shape: (4, depth, 2) where last dim is [RY, RZ].
            qubits (list[int]): List of 4 qubit indices forming a 2x2 window.
        """
        n = len(qubits)
        if n != 4:
            raise ValueError("Convolution window must be exactly 4 qubits (2x2) for this kernel.")

        params = np.array(params, dtype=float)

        # Backward-compat handling:
        # If user passes a flat vector of length 4 * n (old API), reshape conservatively to depth=1, 2 angles.
        # Old code expected length = 4 * n with 4 layers; we now map to (4, 1, 2) by slicing the first 8 values.
        if params.ndim == 1:
            if params.size < 8:
                raise ValueError("Flat params must have at least 8 values (4 qubits * [RY,RZ]).")
            params = params[:8].reshape(4, 1, 2)
        elif params.ndim == 2:
            # Allow (4, 2) -> interpret as single depth with [RY,RZ] per qubit
            if params.shape == (4, 2):
                params = params.reshape(4, 1, 2)
            else:
                raise ValueError("2D params must be shape (4,2) = [RY,RZ] per qubit.")
        elif params.ndim == 3:
            if params.shape[0] != 4 or params.shape[2] != 2:
                raise ValueError("3D params must be (4, depth, 2) with last dim [RY,RZ].")
        else:
            raise ValueError("Unsupported params shape; use (4, depth, 2) or flat length >= 8.")

        depth = params.shape[1]

        # Single-qubit rotations per depth slice
        for d in range(depth):
            for i, q in enumerate(qubits):
                theta_ry = float(params[i, d, 0])
                theta_rz = float(params[i, d, 1])
                qml.RY(theta_ry, wires=q)
                qml.RZ(theta_rz, wires=q)

            # Local entanglement inside the 2x2 window.
            # Window layout indices:
            # [ q0 q1
            #   q2 q3 ]
            q0, q1, q2, q3 = qubits

            # Horizontal edges
            qml.CNOT(wires=[q0, q1])
            qml.CNOT(wires=[q2, q3])
            # Vertical edges
            qml.CNOT(wires=[q0, q2])
            qml.CNOT(wires=[q1, q3])
            # Optional diagonal to mix but remain shallow
            qml.CNOT(wires=[q0, q3])

    @staticmethod
    def get_kernel_param_count(num_qubits: int, depth: int = 1) -> int:
        """Number of parameters for a 2x2 window kernel: 4 qubits * depth * 2 (RY,RZ)."""
        if num_qubits != 4:
            raise ValueError("This kernel is defined for a 2x2 window (4 qubits).")
        return 4 * depth * 2

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
