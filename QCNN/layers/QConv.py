import numpy as np
import pennylane as qml


class QuantumNativeConvolution:
    """
    Quantum-native 2×2 convolution operator.
    Implements a shared parameterized quantum kernel applied across
    local 2×2 windows in the qubit grid, using RY/RZ rotations and
    shallow entanglement consistent with NISQ constraints.
    """

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

        # Accept autograd types; avoid forcing dtype=float here
        params_arr = np.asarray(params)

        # Backward-compat handling and shape normalization:
        # Normalize input to shape (4, depth, 2) with last dim [RY, RZ]
        if params_arr.ndim == 1:
            # Flat vector -> must be at least 8 entries (4 qubits * [RY,RZ])
            if params_arr.size < 8:
                raise ValueError("Flat params must have at least 8 values (4 qubits * [RY,RZ]).")
            params_arr = params_arr[:8].reshape(4, 1, 2)
        elif params_arr.ndim == 2:
            # Allow (4, 2) -> interpret as single depth with [RY,RZ] per qubit
            if params_arr.shape == (4, 2):
                params_arr = params_arr.reshape(4, 1, 2)
            else:
                raise ValueError("2D params must be shape (4,2) = [RY,RZ] per qubit.")
        elif params_arr.ndim == 3:
            if params_arr.shape[0] != 4 or params_arr.shape[2] != 2:
                raise ValueError("3D params must be (4, depth, 2) with last dim [RY,RZ].")
        else:
            raise ValueError("Unsupported params shape; use (4, depth, 2) or flat length >= 8.")

        depth = params_arr.shape[1]

        # Single-qubit rotations per depth slice
        for d in range(depth):
            for i, q in enumerate(qubits):
                theta_ry = params_arr[i, d, 0]
                theta_rz = params_arr[i, d, 1]
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

        # No data-dependent rescaling; stability and weight sharing come from reusing the same 'params'
        # for every 2x2 window within a given convolutional layer.

    @staticmethod
    def get_kernel_param_count(num_qubits: int, depth: int = 1) -> int:
        """
        Returns the number of trainable parameters for a single 2×2 quantum kernel.
        
        For a window of 4 qubits:
            RY and RZ per qubit per depth → 4 * depth * 2.

        Args:
            num_qubits: number of qubits in window (must be 4)
            depth: number of rotation layers

        Returns:
            Total parameter count for kernel.
        """
        if num_qubits != 4:
            raise ValueError("This kernel is defined for a 2x2 window (4 qubits).")
        return 4 * depth * 2

    @staticmethod
    def get_conv_windows(image_size: int) -> list[list[int]]:
        """
        Generates all 2×2 convolution windows for an image-size × image-size qubit grid.
        Windows are returned as lists of 4 qubit indices in raster-scan order.

        Args:
            image_size: width/height of square grid

        Returns:
            List of 4-qubit windows (as index lists) for sliding 2×2 convolution.
        """
        windows = []
        for row in range(image_size - 1):
            for col in range(image_size - 1):
                top_left = row * image_size + col
                top_right = row * image_size + (col + 1)
                bottom_left = (row + 1) * image_size + col
                bottom_right = (row + 1) * image_size + (col + 1)
                windows.append([top_left, top_right, bottom_left, bottom_right])
        return windows
