from .QCNNModel import PureQuantumNativeCNN

__all__ = ["PureQuantumNativeCNN"]

AVAILABLE_MODELS = {
    "pure_quantum_cnn": PureQuantumNativeCNN
}

def create_model(model_type="pure_quantum_cnn", config=None):
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if config is None:
        from ..config import get_default_config
        config = get_default_config()
    
    return AVAILABLE_MODELS[model_type](config)

def get_available_models():
    return list(AVAILABLE_MODELS.keys())
