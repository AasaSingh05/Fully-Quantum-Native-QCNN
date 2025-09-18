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
        # Layer 1: Individual qubit rotations
        for i, feature in enumerate(data[:len(wires)]):
            qml.Hadamard(wires=wires[i])
            qml.RZ(feature * np.pi, wires=wires[i])
        
        # Layer 2: Quantum entangling interactions  
        for i in range(len(wires)):
            for j in range(i + 1, len(wires)):
                if i < len(data) and j < len(data):
                    # ZZ interaction encoding feature correlations
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(data[i] * data[j] * np.pi/4, wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])
        
        # Layer 3: Second-order rotations
        for i, feature in enumerate(data[:len(wires)]):
            qml.RY(feature * np.pi / 2, wires=wires[i])