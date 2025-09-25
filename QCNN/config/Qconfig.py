class QuantumNativeConfig:
    """Configuration ensuring purely quantum operations"""
    
    def __init__(self):
        # Data parameters
        self.image_size = 4  # 4x4 = 16 qubits
        self.n_qubits = self.image_size ** 2
        
        # Pure quantum architecture parameters  
        self.n_conv_layers = 4 #2
        self.kernel_size = 2  # 2x2 quantum kernels
        self.encoding_type = 'feature_map'  # Quantum data encoding
        
        # Quantum training parameters
        self.learning_rate = 0.005
        self.n_epochs = 100 #80 
        self.batch_size = 16 #16
        
        # Quantum device
        self.device = 'lightning.qubit'  
        self.shots = None  # Exact quantum simulation


config = QuantumNativeConfig()
print(f"Configuration: {config.n_qubits} qubits, {config.n_conv_layers} quantum layers")
