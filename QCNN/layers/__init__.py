from .QConv import QuantumNativeConvolution
from .QPool import QuantumNativePooling
from .QuanvLayer import QuanvolutionalLayer

__all__ = [
    "QuantumNativeConvolution",
    "QuantumNativePooling",
    "QuanvolutionalLayer"
]

LAYER_TYPES = {
    "convolution": QuantumNativeConvolution,
    "pooling": QuantumNativePooling,
    "quanvolutional": QuanvolutionalLayer
}

def get_conv_layer():
    return QuantumNativeConvolution()

def get_pooling_layer():
    return QuantumNativePooling()

def get_kernel_param_count():
    return QuantumNativeConvolution.get_kernel_param_count()
