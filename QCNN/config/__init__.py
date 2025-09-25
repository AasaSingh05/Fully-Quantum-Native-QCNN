from .Qconfig import QuantumNativeConfig

__all__ = ["QuantumNativeConfig"]

DEFAULT_CONFIG = QuantumNativeConfig()

def get_default_config():
    return QuantumNativeConfig()

def create_custom_config(image_size=4, n_conv_layers=2, encoding_type='amplitude'):
    config = QuantumNativeConfig()
    config.image_size = image_size
    config.n_qubits = image_size ** 2
    config.n_conv_layers = n_conv_layers
    config.encoding_type = encoding_type
    return config