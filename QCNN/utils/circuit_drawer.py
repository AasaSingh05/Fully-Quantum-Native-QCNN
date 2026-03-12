#!/usr/bin/env python3
"""
Utility script to generate and save circuit diagrams for the Pure Quantum Native QCNN.
Visualizes the encoding, convolutional kernels, pooling stages, and classifier head.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Try importing dependencies
try:
    import pennylane as qml
    from QCNN.config.Qconfig import QuantumNativeConfig
    from QCNN.models.QCNNModel import PureQuantumNativeCNN
    from QCNN.encoding.QEncoder import PureQuantumEncoder
    from QCNN.layers.QConv import QuantumNativeConvolution
    from QCNN.layers.QPool import QuantumNativePooling
except ImportError as e:
    print(f"Error: Missing dependencies or incorrect project structure. {e}")
    sys.exit(1)

def draw_circuits(output_dir="Results/Graphs/Circuit_Architecture", encoding='feature_map'):
    """
    Instantiates a representative QCNN model and saves its circuit diagrams.
    Generates both isolated component diagrams and the full architecture.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Config and Model
    config = QuantumNativeConfig.from_image_size(4, encoding) # 4x4 or similar
    config.n_qubits = 8
    model = PureQuantumNativeCNN(config)
    
    params = model.quantum_params
    
    print(f"Generating diagrams for {config.n_qubits}-qubit QCNN ({encoding} encoding)...")
    
    # Helpers for plotting
    def save_qnode_diagram(qnode, inputs, title, filename):
        try:
            fig, ax = qml.draw_mpl(qnode, decimals=2)(*inputs)
            plt.title(title)
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [SUCCESS] {title} saved to: {save_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to draw {title}: {e}")

    # ==========================================
    # 1. ENCODING CIRCUIT
    # ==========================================
    dev_enc = qml.device('default.qubit', wires=8)
    @qml.qnode(dev_enc)
    def enc_circuit(x):
        if encoding in ('amplitude', 'patch'):
            PureQuantumEncoder.amplitude_encoding(x, list(range(8)))
        else:
            PureQuantumEncoder.quantum_feature_map(x, list(range(8)))
        return qml.expval(qml.PauliZ(0))
    
    x_enc = np.random.uniform(0, 1, 2**8) if encoding in ('amplitude', 'patch') else np.random.uniform(0, 2*np.pi, 8)
    if encoding in ('amplitude', 'patch'): x_enc /= np.linalg.norm(x_enc)
    save_qnode_diagram(enc_circuit, (x_enc,), f"Encoding Layer ({encoding.capitalize()})", f"component_1_encoding_{encoding}.png")

    # ==========================================
    # 2. CONVOLUTION KERNEL (2x2 Window)
    # ==========================================
    dev_conv = qml.device('default.qubit', wires=4)
    @qml.qnode(dev_conv)
    def conv_circuit(p):
        QuantumNativeConvolution.quantum_conv2d_kernel(p, [0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))
    
    save_qnode_diagram(conv_circuit, (params['quantum_conv_kernel_0'],), "Convolutional Kernel (SU(2), Depth 4)", "component_2_convolution_kernel.png")

    # ==========================================
    # 3. POOLING KERNEL (2 Qubits -> 1)
    # ==========================================
    dev_pool = qml.device('default.qubit', wires=2)
    @qml.qnode(dev_pool)
    def pool_circuit(p):
        QuantumNativePooling.quantum_unitary_pooling(p, input_qubits=[0], output_qubits=[1])
        return qml.expval(qml.PauliZ(0))
    
    save_qnode_diagram(pool_circuit, (params['quantum_pooling_0'][:3],), "Pooling Operation", "component_3_pooling_kernel.png")

    # ==========================================
    # 4. CLASSIFIER HEAD (Depth 2 SU(2) on 4 active qubits)
    # ==========================================
    dev_cls = qml.device('default.qubit', wires=4)
    @qml.qnode(dev_cls)
    def classifier_circuit(p):
        active_qubits = [0, 1, 2, 3]
        n_active = len(active_qubits)
        readout = active_qubits[0]

        # Layer 1
        for i, q in enumerate(active_qubits[:min(n_active, 4)]):
            qml.RX(p[i * 2 % 32], wires=q)
            qml.RY(p[(i * 2 + 1) % 32], wires=q)
            qml.RZ(p[(i * 2 + 8) % 32], wires=q)

        # Entanglement
        for i in range(n_active - 1):
            qml.CNOT(wires=[active_qubits[i], active_qubits[i+1]])
        if n_active >= 2:
            qml.CNOT(wires=[active_qubits[n_active-1], active_qubits[0]])

        # Layer 2
        for i, q in enumerate(active_qubits[:min(n_active, 4)]):
            qml.RX(p[(i * 2 + 16) % 32], wires=q)
            qml.RY(p[(i * 2 + 17) % 32], wires=q)

        if n_active >= 2:
            qml.CNOT(wires=[active_qubits[0], active_qubits[min(n_active-1, 1)]])
        qml.RZ(p[31], wires=readout)
        
        return qml.expval(qml.PauliZ(readout))
    
    save_qnode_diagram(classifier_circuit, (params['quantum_classifier'],), "Deep Variational Classifier (32 Params)", "component_4_classifier_head.png")

    # ==========================================
    # 5. FULL FORWARD CIRCUIT
    # ==========================================
    try:
        params_flat = model._flatten_params(params)
        fig, ax = qml.draw_mpl(model.quantum_circuit, decimals=2)(x_enc, params_flat)
        plt.title(f"Full QCNN Architecture ({encoding.capitalize()} Encoding)")
        
        save_path = os.path.join(output_dir, f"full_architecture_{encoding}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [SUCCESS] Full architecture saved to: {save_path}")
        
        # Save text-based version
        text_path = os.path.join(output_dir, f"full_architecture_{encoding}.txt")
        with open(text_path, "w") as f:
            f.write(qml.draw(model.quantum_circuit)(x_enc, params_flat))
    except Exception as e:
        print(f"  [ERROR] Failed to generate full plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QCNN Circuit Diagrams')
    parser.add_argument('--output', type=str, default='Results/Graphs/Circuit_Architecture',
                        help='Output directory for diagrams')
    parser.add_argument('--encoding', type=str, default='feature_map',
                        choices=['feature_map', 'amplitude', 'patch'],
                        help='Encoding type to visualize')
    
    args = parser.parse_args()
    
    print("="*50)
    print(" QCNN CIRCUIT DIAGRAM GENERATOR")
    print("="*50)
    
    draw_circuits(args.output, args.encoding)
    
    print("\nDiagram generation complete.")
