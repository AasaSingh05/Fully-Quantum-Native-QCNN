from .QEncoder import PureQuantumEncoder

__all__ = ["PureQuantumEncoder"]

ENCODING_METHODS = [
    "amplitude_encoding",
    "quantum_feature_map"
]

def get_available_encodings():
    return ENCODING_METHODS.copy()

def get_encoder():
    return PureQuantumEncoder()