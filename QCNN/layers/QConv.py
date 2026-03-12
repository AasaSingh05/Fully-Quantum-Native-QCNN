import pennylane as qml
import pennylane.numpy as pnp
import numpy as np


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
        params_arr = pnp.asarray(params)

        # Backward-compat handling and shape normalization:
        # Normalize input to shape (4, depth, 3) with last dim [RX, RY, RZ]
        if params_arr.ndim == 1:
            # Flat vector -> [4 qubits * depth * 3]
            depth = params_arr.size // 12
            if depth == 0:
                raise ValueError("Flat params too short for SU(2) kernel.")
            params_arr = params_arr[:depth*12].reshape(4, depth, 3)
        elif params_arr.ndim == 2:
            # (4, 3) -> interpret as depth 1
            if params_arr.shape == (4, 3):
                params_arr = params_arr.reshape(4, 1, 3)
            else:
                raise ValueError("2D params must be shape (4,3) for SU(2).")
        elif params_arr.ndim == 3:
            if params_arr.shape[0] != 4 or params_arr.shape[2] != 3:
                raise ValueError("3D params must be (4, depth, 3) for SU(2).")
        else:
            raise ValueError("Unsupported params shape; use (4, depth, 3).")

        depth = params_arr.shape[1]

        # Single-qubit rotations per depth slice
        for d in range(depth):
            for i, q in enumerate(qubits):
                qml.RX(params_arr[i, d, 0], wires=q)
                qml.RY(params_arr[i, d, 1], wires=q)
                qml.RZ(params_arr[i, d, 2], wires=q)

            # All-to-all entanglement inside the 2x2 window.
            # Window layout: [ q0 q1, q2 q3 ]
            q0, q1, q2, q3 = qubits

            # Horizontal
            qml.CNOT(wires=[q0, q1])
            qml.CNOT(wires=[q2, q3])
            # Vertical
            qml.CNOT(wires=[q0, q2])
            qml.CNOT(wires=[q1, q3])
            # Diagonals (all-to-all)
            qml.CNOT(wires=[q0, q3])
            qml.CNOT(wires=[q1, q2])

        # No data-dependent rescaling; stability and weight sharing come from reusing the same 'params'
        # for every 2x2 window within a given convolutional layer.

    @staticmethod
    def get_kernel_param_count(num_qubits: int, depth: int = 1) -> int:
        """
        Returns the number of trainable parameters for a single 2×2 quantum kernel.
        
        For a window of 4 qubits:
            RX, RY and RZ per qubit per depth → 4 * depth * 3.

        Args:
            num_qubits: number of qubits in window (must be 4)
            depth: number of rotation layers

        Returns:
            Total parameter count for kernel.
        """
        if num_qubits != 4:
            raise ValueError("This kernel is defined for a 2x2 window (4 qubits).")
        return 4 * depth * 3

    @staticmethod
    def get_conv_windows(width: int, height: int = None) -> list[list[int]]:
        """
        Generates all 2×2 convolution windows for a width × height qubit grid.
        Windows are returned as lists of 4 qubit indices in raster-scan order.

        Args:
            width: width of the grid
            height: height of the grid (defaults to width if square)

        Returns:
            List of 4-qubit windows (as index lists) for sliding 2×2 convolution.
        """
        if height is None:
            height = width
            
        windows = []
        for row in range(height - 1):
            for col in range(width - 1):
                top_left = row * width + col
                top_right = row * width + (col + 1)
                bottom_left = (row + 1) * width + col
                bottom_right = (row + 1) * width + (col + 1)
                windows.append([top_left, top_right, bottom_left, bottom_right])
        return windows
