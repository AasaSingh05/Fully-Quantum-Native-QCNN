import math


class QuantumNativeConfig:
    """Configuration ensuring purely quantum operations"""
    
    def __init__(self):
        # Data parameters
        self.image_size = 4  # 4x4 = 16 features
        self.n_qubits = self.image_size ** 2  # Default: 1 qubit per pixel
        
        # Dataset parameters
        self.n_classes = 2  # Binary classification
        self.n_features = self.image_size ** 2  # Total input features
        self.preprocessing_mode = 'minmax'  # 'minmax', 'standard', 'robust'
        
        # Encoding strategy: 'feature_map', 'amplitude', 'patch'
        #   feature_map  – 1 qubit per feature (original, image_size² qubits)
        #   amplitude    – log₂(features) qubits via amplitude embedding
        #   patch        – quanvolutional preprocessing then QCNN on reduced map
        self.encoding_type = 'feature_map'
        
        # Pure quantum architecture parameters  
        self.n_conv_layers = 4
        self.kernel_size = 2  # 2x2 quantum kernels
        
        # Quanvolutional layer parameters (used when encoding_type == 'patch')
        self.patch_size = 4        # Patch width/height for quanvolutional filter
        self.patch_stride = 4      # Stride between patches (== patch_size → non-overlapping)
        self.n_quanv_filters = 4   # Number of random quantum filters per patch
        self.quanv_qubits = 16     # Qubits used per patch circuit (patch_size²)
        
        # Quantum training parameters
        self.learning_rate = 0.005
        self.n_epochs = 100
        self.batch_size = 16
        
        # Quantum device
        self.device = 'lightning.qubit'  
        self.shots = None  # Exact quantum simulation
    
    def configure_for_image(self, image_size: int, encoding: str = 'auto'):
        """
        Auto-configure qubit count and encoding based on image size.
        
        Args:
            image_size: Width/height of square image
            encoding: 'auto', 'feature_map', 'amplitude', or 'patch'
        """
        self.image_size = image_size
        self.n_features = image_size ** 2
        
        if encoding == 'auto':
            if image_size <= 4:
                encoding = 'feature_map'
            elif image_size <= 16:
                encoding = 'amplitude'
            else:
                encoding = 'patch'
        
        self.encoding_type = encoding
        
        if encoding == 'feature_map':
            # 1 qubit per pixel
            self.n_qubits = self.n_features
            if self.n_qubits > 25:
                raise ValueError(
                    f"Cannot simulate {self.n_qubits} qubits via 'feature_map' encoding. "
                    "Classical simulation is bounded to ~25 qubits. "
                    "Use 'amplitude' or 'patch' encoding, or downsample the image."
                )
        
        elif encoding == 'amplitude':
            # log₂(features) qubits – pad features to nearest power of 2
            n_features_padded = 2 ** math.ceil(math.log2(max(self.n_features, 2)))
            self.n_qubits = math.ceil(math.log2(n_features_padded))
            self.n_features = n_features_padded
        
        elif encoding == 'patch':
            # Quanvolutional: QCNN operates on the reduced feature map
            self.quanv_qubits = self.patch_size ** 2
            out_size = (image_size - self.patch_size) // self.patch_stride + 1
            reduced_features = out_size * out_size * self.n_quanv_filters
            # QCNN runs on the reduced map using amplitude encoding
            n_padded = 2 ** math.ceil(math.log2(max(reduced_features, 2)))
            self.n_qubits = math.ceil(math.log2(n_padded))
            self.n_features = reduced_features
        
        # Adjust conv layers based on qubit count
        if self.n_qubits < 8:
            self.n_conv_layers = min(self.n_conv_layers, 3)
        if self.n_qubits < 4:
            self.n_conv_layers = min(self.n_conv_layers, 2)
        
        return self
    
    @classmethod
    def from_image_size(cls, image_size: int, encoding: str = 'auto'):
        """
        Factory: create a config auto-tuned for a given image size.
        
        Args:
            image_size: Width/height of square images
            encoding: 'auto', 'feature_map', 'amplitude', or 'patch'
        
        Returns:
            Configured QuantumNativeConfig instance
        """
        config = cls()
        config.configure_for_image(image_size, encoding)
        return config


config = QuantumNativeConfig()
print(f"Configuration: {config.n_qubits} qubits, {config.n_conv_layers} quantum layers")
