# Quantum-Native QCNN  
A Fully Quantum-Convolutional Neural Network Built with PennyLane

This repository implements a **fully quantum-native convolutional neural network (QCNN)** that performs **encoding, convolution, pooling, and classification entirely using quantum circuits**, with no classical convolutional operations at any stage.

The current prototype operates on **4×4 images mapped onto a 16-qubit grid**, using translationally shared quantum kernels, unitary pooling layers, and a variational quantum head. All training is performed using differentiable quantum circuits and parameter-shift gradients.

This project is part of a broader research effort toward **quantum-native deep learning architectures** suitable for near-term quantum devices.

---

## 1. What This Project Implements

The QCNN reproduces the structure of classical CNNs using **only quantum operations**:

- **Feature Encoding** via angle-based quantum feature maps  
- **Quantum Convolution** using weight-shared 2×2 local quantum kernels  
- **Quantum Pooling** via trainable 2-qubit unitary reductions  
- **Quantum Classification Head** producing a scalar prediction ⟨Z⟩  
- **End-to-end Differentiability** through PennyLane's autodiff and parameter-shift rule

There is **no hybrid classical convolution path**.  
Every layer (conv, pool, head) is a quantum circuit.

---

## 2. Repository Structure

