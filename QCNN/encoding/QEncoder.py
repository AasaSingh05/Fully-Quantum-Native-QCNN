import numpy as np
import pennylane as qml


class PureQuantumEncoder:
    """
    Quantum-native encoder implementing amplitude encoding and
    rotation-based quantum feature maps. Provides purely unitary
    data embedding for QCNN pipelines without classical preprocessing.
    """

    @staticmethod
    def amplitude_encoding(data: np.ndarray, wires: list[int]) -> None:
        """
        Pure quantum amplitude encoding with batch support.
        Maps classical vectors into the amplitudes of an n-qubit
        quantum state, padding or truncating as required. 
        
        Args:
            data: 1D array (features,) or 2D array (batch_size, features)
            wires: list of qubit indices for the embedding
        """
        data = np.asarray(data)
        is_batched = data.ndim == 2
        max_size = 2 ** len(wires)
        
        # Determine current feature size
        feat_size = data.shape[1] if is_batched else data.shape[0]
        
        if feat_size > max_size:
            if is_batched:
                data = data[:, :max_size]
            else:
                data = data[:max_size]
        elif feat_size < max_size:
            if is_batched:
                padded = np.zeros((data.shape[0], max_size))
                padded[:, :feat_size] = data
                data = padded
            else:
                padded = np.zeros(max_size)
                padded[:feat_size] = data
                data = padded
        
        # Normalize for quantum amplitudes (must sum to 1)
        if is_batched:
            norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-10
            data_norm = data / norms
        else:
            data_norm = data / (np.linalg.norm(data) + 1e-10)
        
        # Pure quantum encoding - natively supports batched input
        qml.AmplitudeEmbedding(features=data_norm, wires=wires, normalize=True)
    
    @staticmethod
    def quantum_feature_map(data: np.ndarray, wires: list[int]) -> None:
        """
        Rotation-based entangling quantum feature map.
        Supports batched inputs.
        
        Args:
            data: 1D array (features,) or 2D array (batch_size, features)
            wires: list of qubit indices used for the feature map
        """
        data = np.asarray(data, dtype=float)
        is_batched = data.ndim == 2
        feat_size = data.shape[1] if is_batched else data.shape[0]
        
        L = min(feat_size, len(wires))
        
        if is_batched:
            x = data[:, :L]
        else:
            x = data[:L]
            
        if x.size > 0:
            x = np.clip(x, -1.0, 1.0)  # preserve sign, bound range

        angles_z = np.pi * x           # RZ(π x) ∈ [-π, π]
        angles_y = 0.5 * np.pi * x     # RY((π/2) x) ∈ [-π/2, π/2]

        # Layer 1: Individual qubit rotations
        for i in range(L):
            qml.Hadamard(wires=wires[i])
            angle_z_i = angles_z[:, i] if is_batched else angles_z[i]
            qml.RZ(angle_z_i, wires=wires[i])

        # Layer 2: Quantum entangling interactions
        for i in range(L - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            if is_batched:
                corr_angle = (x[:, i] * x[:, i + 1]) * (np.pi / 4.0)
            else:
                corr_angle = (x[i] * x[i + 1]) * (np.pi / 4.0)
            qml.RZ(corr_angle, wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])

        # Optional 2D grid-style local entanglement for square layouts (e.g., 4x4)
        if len(wires) >= 4:
            width = int(np.sqrt(len(wires)))
            if width * width == len(wires):
                # Horizontal neighbors
                for r in range(width):
                    for c in range(width - 1):
                        a = r * width + c
                        b = r * width + (c + 1)
                        if a < L and b < L:
                            qml.CNOT(wires=[wires[a], wires[b]])
                            if is_batched:
                                qml.RZ((x[:, a] * x[:, b]) * (np.pi / 8.0), wires=wires[b])
                            else:
                                qml.RZ((x[a] * x[b]) * (np.pi / 8.0), wires=wires[b])
                            qml.CNOT(wires=[wires[a], wires[b]])
                # Vertical neighbors
                for c in range(width):
                    for r in range(width - 1):
                        a = r * width + c
                        b = (r + 1) * width + c
                        if a < L and b < L:
                            qml.CNOT(wires=[wires[a], wires[b]])
                            if is_batched:
                                qml.RZ((x[:, a] * x[:, b]) * (np.pi / 8.0), wires=wires[b])
                            else:
                                qml.RZ((x[a] * x[b]) * (np.pi / 8.0), wires=wires[b])
                            qml.CNOT(wires=[wires[a], wires[b]])

        # Layer 3: Second-order rotations
        for i in range(L):
            angle_y_i = angles_y[:, i] if is_batched else angles_y[i]
            qml.RY(angle_y_i, wires=wires[i])
