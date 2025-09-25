import numpy as np
import pennylane as qml


class PureQuantumEncoder:    
    @staticmethod
    def amplitude_encoding(data: np.ndarray, wires: list[int]) -> None:
        """
        Pure quantum amplitude encoding
        Encodes 2^n classical values into n qubits via quantum amplitudes
        """
        # Ensure data fits in quantum state space
        max_size = 2 ** len(wires)
        
        if len(data) > max_size:
            data = data[:max_size]
        elif len(data) < max_size:
            # Pad with zeros
            padded_data = np.zeros(max_size)
            padded_data[:len(data)] = data
            data = padded_data
        
        # Normalize for quantum amplitudes (must sum to 1)
        data_norm = data / (np.linalg.norm(data) + 1e-10)
        
        # Pure quantum encoding - no classical preprocessing
        qml.AmplitudeEmbedding(features=data_norm, wires=wires, normalize=True)
    
    @staticmethod  
    def quantum_feature_map(data: np.ndarray, wires: list[int]) -> None:
        """
        Advanced quantum feature map with entangling layers
        Creates quantum correlations impossible classically
        """
        # Symmetric scheme: use one feature per wire, clip to [-1, 1],
        # then map to bounded angles preserving sign information
        L = min(len(data), len(wires))
        x = np.asarray(data[:L], dtype=float)
        if x.size > 0:
            x = np.clip(x, -1.0, 1.0)  # preserve sign, bound range

        angles_z = np.pi * x           # RZ(π x) ∈ [-π, π]
        angles_y = 0.5 * np.pi * x     # RY((π/2) x) ∈ [-π/2, π/2]

        # Layer 1: Individual qubit rotations
        for i in range(L):
            qml.Hadamard(wires=wires[i])
            qml.RZ(angles_z[i], wires=wires[i])

        # Layer 2: Quantum entangling interactions  
        # Sparse local entanglement (linear chain) to keep depth manageable
        for i in range(L - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
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
                            qml.RZ((x[a] * x[b]) * (np.pi / 8.0), wires=wires[b])
                            qml.CNOT(wires=[wires[a], wires[b]])
                # Vertical neighbors
                for c in range(width):
                    for r in range(width - 1):
                        a = r * width + c
                        b = (r + 1) * width + c
                        if a < L and b < L:
                            qml.CNOT(wires=[wires[a], wires[b]])
                            qml.RZ((x[a] * x[b]) * (np.pi / 8.0), wires=wires[b])
                            qml.CNOT(wires=[wires[a], wires[b]])

        # Layer 3: Second-order rotations
        for i in range(L):
            qml.RY(angles_y[i], wires=wires[i])
