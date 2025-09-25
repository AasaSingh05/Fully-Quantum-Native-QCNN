from .QConv import QuantumNativeConvolution
from .QPool import QuantumNativePooling

__all__ = [
    "QuantumNativeConvolution",
    "QuantumNativePooling"
]

LAYER_TYPES = {
    "convolution": QuantumNativeConvolution,
    "pooling": QuantumNativePooling
}

def get_conv_layer():
    return QuantumNativeConvolution()

def get_pooling_layer():
    return QuantumNativePooling()

def get_kernel_param_count():
    return QuantumNativeConvolution.get_kernel_param_count()
