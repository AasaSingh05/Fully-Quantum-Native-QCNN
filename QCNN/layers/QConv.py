import numpy as np
import pennylane as qml

class QuantumNativeConvolution:
    """Pure quantum convolutional operations"""
    
    @staticmethod
    def quantum_conv2d_kernel(params: np.ndarray, qubits:list[int]) -> None:
        """
        Pure quantum 2x2 convolutional kernel
        Creates local quantum entanglement for feature extraction
        No classical convolution operations
        """
        if len(qubits) != 4:
            raise ValueError("Quantum 2x2 kernel requires exactly 4 qubits")
            
        # Quantum convolution structure:
        # [q0] -- [q1]
        #  |       |  
        # [q2] -- [q3]
        
        # Layer 1: Individual quantum rotations (like classical conv weights)
        for i, qubit in enumerate(qubits):
            qml.RY(params[i], wires=qubit)
            qml.RZ(params[i + 4], wires=qubit)
        
        # Layer 2: Nearest-neighbor quantum entanglement
        qml.CNOT(wires=[qubits[0], qubits[1]])  # Top edge
        qml.CNOT(wires=[qubits[2], qubits[3]])  # Bottom edge  
        qml.CNOT(wires=[qubits[0], qubits[2]])  # Left edge
        qml.CNOT(wires=[qubits[1], qubits[3]])  # Right edge
        
        # Layer 3: More quantum rotations
        for i, qubit in enumerate(qubits):
            qml.RX(params[i + 8], wires=qubit)
        
        # Layer 4: Diagonal quantum entanglement (impossible classically)
        qml.CNOT(wires=[qubits[0], qubits[3]])  # Main diagonal
        qml.CNOT(wires=[qubits[1], qubits[2]])  # Anti-diagonal
        
        # Layer 5: Final quantum rotations  
        for i, qubit in enumerate(qubits):
            qml.RY(params[i + 12], wires=qubit)
    
    @staticmethod
    def get_kernel_param_count() -> int:
        """Number of parameters in quantum conv kernel"""
        return 16  # 4 layers × 4 qubits
    
    @staticmethod
    def get_conv_windows(image_size: int) ->list[list[int]]:
        """
        Generate all 2x2 sliding windows for quantum convolution
        Returns qubit indices for each window
        """
        windows = []
        for row in range(image_size - 1):
            for col in range(image_size - 1):
                # Map 2D image coordinates to 1D qubit indices
                top_left = row * image_size + col
                top_right = row * image_size + (col + 1)  
                bottom_left = (row + 1) * image_size + col
                bottom_right = (row + 1) * image_size + (col + 1)
                
                windows.append([top_left, top_right, bottom_left, bottom_right])
        
        return windows
