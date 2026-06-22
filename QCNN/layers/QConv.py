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

    # Rotation gates per supported kernel rotation set, in fixed order.
    _ROTATION_GATES = {
        'su2': ('RX', 'RY', 'RZ'),
        'ry': ('RY',),
    }

    @staticmethod
    def rotations_per_qubit(rotations: str = 'su2') -> int:
        """Number of single-qubit rotation params per qubit for a rotation set."""
        try:
            return len(QuantumNativeConvolution._ROTATION_GATES[rotations])
        except KeyError:
            raise ValueError(f"Unknown kernel_rotations '{rotations}'. "
                             f"Use one of {list(QuantumNativeConvolution._ROTATION_GATES)}.")

    @staticmethod
    def quantum_conv2d_kernel(params: np.ndarray, qubits: list[int],
                              rotations: str = 'su2', entanglement: str = 'full') -> None:
        """
        Quantum convolution kernel for a 2x2 window (4 qubits).
        Applies parameterized rotations with local entanglement.

        Args:
            params (np.ndarray): Trainable kernel parameters for a single 2x2 window.
                                 Shape: (4, depth, R) where R is the rotation count
                                 (3 for 'su2' → [RX, RY, RZ], 1 for 'ry' → [RY]).
            qubits (list[int]): List of 4 qubit indices forming a 2x2 window.
            rotations (str): 'su2' (RX,RY,RZ) or 'ry' (RY only). Ablation #4.
            entanglement (str): 'full' (6 CNOTs), 'one_diagonal' (5 CNOTs,
                                paper-faithful), or 'none'. Ablation #4.
        """
        n = len(qubits)
        if n != 4:
            raise ValueError("Convolution window must be exactly 4 qubits (2x2) for this kernel.")

        gates = QuantumNativeConvolution._ROTATION_GATES.get(rotations)
        if gates is None:
            raise ValueError(f"Unknown kernel_rotations '{rotations}'.")
        rpq = len(gates)

        # Accept autograd types; avoid forcing dtype=float here
        params_arr = pnp.asarray(params)

        # Normalize input to shape (4, depth, rpq).
        if params_arr.ndim == 1:
            depth = params_arr.size // (4 * rpq)
            if depth == 0:
                raise ValueError("Flat params too short for the requested kernel.")
            params_arr = params_arr[:depth * 4 * rpq].reshape(4, depth, rpq)
        elif params_arr.ndim == 2:
            if params_arr.shape == (4, rpq):
                params_arr = params_arr.reshape(4, 1, rpq)
            else:
                raise ValueError(f"2D params must be shape (4,{rpq}) for '{rotations}'.")
        elif params_arr.ndim == 3:
            if params_arr.shape[0] != 4 or params_arr.shape[2] != rpq:
                raise ValueError(f"3D params must be (4, depth, {rpq}) for '{rotations}'.")
        else:
            raise ValueError("Unsupported params shape.")

        depth = params_arr.shape[1]

        # Single-qubit rotations per depth slice
        for d in range(depth):
            for i, q in enumerate(qubits):
                for r, gate in enumerate(gates):
                    getattr(qml, gate)(params_arr[i, d, r], wires=q)

            # Intra-window entanglement. Window layout: [ q0 q1, q2 q3 ]
            q0, q1, q2, q3 = qubits
            if entanglement != 'none':
                # Edges (always present when entangling).
                qml.CNOT(wires=[q0, q1])  # horizontal
                qml.CNOT(wires=[q2, q3])
                qml.CNOT(wires=[q0, q2])  # vertical
                qml.CNOT(wires=[q1, q3])
                # Diagonals.
                qml.CNOT(wires=[q0, q3])  # one diagonal (paper-faithful minimum)
                if entanglement == 'full':
                    qml.CNOT(wires=[q1, q2])  # second diagonal → all-to-all
                elif entanglement != 'one_diagonal':
                    raise ValueError(f"Unknown conv_entanglement '{entanglement}'.")

        # No data-dependent rescaling; stability and weight sharing come from reusing the same 'params'
        # for every 2x2 window within a given convolutional layer.

    @staticmethod
    def get_kernel_param_count(num_qubits: int, depth: int = 1, rotations: str = 'su2') -> int:
        """
        Returns the number of trainable parameters for a single 2×2 quantum kernel.

        For a window of 4 qubits: (rotations/qubit) per qubit per depth →
        4 * depth * rotations_per_qubit (3 for 'su2', 1 for 'ry').

        Args:
            num_qubits: number of qubits in window (must be 4)
            depth: number of rotation layers
            rotations: rotation set ('su2' or 'ry')

        Returns:
            Total parameter count for kernel.
        """
        if num_qubits != 4:
            raise ValueError("This kernel is defined for a 2x2 window (4 qubits).")
        return 4 * depth * QuantumNativeConvolution.rotations_per_qubit(rotations)

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
