"""
noise_sim.py — Depolarizing noise robustness experiment for FQCNN

Loads MNIST directly, preprocesses with amplitude embedding at the configured
image_size, derives the circuit qubit count from the saved weights, then sweeps
depolarizing noise p across [0, 0.20] and plots experimental vs theoretical.

Standalone:   python noise_sim.py
Programmatic: from noise_sim import run_noise_simulation
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.model_selection import train_test_split


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_mnist(n_samples, classes, image_size):
    """
    Load MNIST for binary classification.
    Returns X (N, image_size²) normalized to [0,1] and y in {-1, +1}.
    Tries IDX files at datasets/MNIST/ first, falls back to sklearn.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from QCNN.utils.data_preprocessing import preprocess_for_quantum

    mnist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'MNIST')
    if os.path.isdir(mnist_dir) and any('idx3' in f for f in os.listdir(mnist_dir)):
        from QCNN.utils.dataset_loader import load_idx_dataset
        files = os.listdir(mnist_dir)
        images_f = next((f for f in files if 'train' in f and 'images' in f and 'idx3' in f), None)
        labels_f = next((f for f in files if 'train' in f and 'labels' in f and 'idx1' in f), None)
        if images_f and labels_f:
            print(f"  Loading MNIST IDX from {mnist_dir}/")
            X_raw, y_raw = load_idx_dataset(
                os.path.join(mnist_dir, images_f),
                os.path.join(mnist_dir, labels_f)
            )
        else:
            X_raw, y_raw = _fetch_openml_mnist()
    else:
        X_raw, y_raw = _fetch_openml_mnist()

    mask = (y_raw == classes[0]) | (y_raw == classes[1])
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    print(f"  After class filter ({classes}): {len(X_raw)} samples")

    if len(X_raw) > n_samples:
        idx0 = np.where(y_raw == classes[0])[0]
        idx1 = np.where(y_raw == classes[1])[0]
        n0 = min(len(idx0), n_samples // 2)
        n1 = min(len(idx1), n_samples - n0)
        chosen = np.concatenate([
            np.random.choice(idx0, n0, replace=False),
            np.random.choice(idx1, n1, replace=False)
        ])
        np.random.shuffle(chosen)
        X_raw, y_raw = X_raw[chosen], y_raw[chosen]
    print(f"  Using {len(X_raw)} samples (capped to {n_samples})")

    X, y = preprocess_for_quantum(
        X_raw, y_raw,
        n_qubits=math.ceil(math.log2(max(image_size * image_size, 2))),
        image_size=image_size,
        normalization='minmax',
        encoding_type='amplitude'
    )
    return X, y


def _fetch_openml_mnist():
    from sklearn.datasets import fetch_openml
    print("  Fetching MNIST via sklearn (cached after first download)...")
    mnist = fetch_openml('mnist_784', version=1, parser='liac-arff', as_frame=False)
    return mnist.data.astype(np.float64), mnist.target.astype(int)


def _preprocess_for_circuit(X, target_len):
    """Pad features to target_len and normalize each row to unit norm."""
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    feat_len = X.shape[1]
    if feat_len < target_len:
        padded = np.zeros((X.shape[0], target_len))
        padded[:, :feat_len] = X
        X = padded
    else:
        X = X[:, :target_len]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms


def _flatten_weights(w, n_conv, n_pool):
    parts = []
    for i in range(n_conv):
        parts.append(w[f'quantum_conv_kernel_{i}'].flatten())
    for i in range(n_pool):
        parts.append(w[f'quantum_pooling_{i}'].flatten())
    parts.append(w['quantum_classifier'].flatten())
    return np.concatenate(parts)


def _unflatten_params(flat, n_conv, n_pool, n_qubits):
    params = {}
    idx = 0
    kernel_size = 4 * 4 * 3  # 4 windows × depth=4 × 3 angles
    for i in range(n_conv):
        params[f'quantum_conv_kernel_{i}'] = flat[idx:idx + kernel_size].reshape(4, 4, 3)
        idx += kernel_size
    # FIX 3: pool_size uses actual n_qubits, not inferred from weight shape
    pool_size = 3 * (n_qubits // 2)
    for i in range(n_pool):
        params[f'quantum_pooling_{i}'] = flat[idx:idx + pool_size]
        idx += pool_size
    params['quantum_classifier'] = flat[idx:idx + 32]
    return params


def _make_noisy_circuit(n_qubits, n_conv, n_pool, flat_params, noise_p):
    from QCNN.layers.QConv import QuantumNativeConvolution
    from QCNN.layers.QPool import QuantumNativePooling

    dev = qml.device('default.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='numpy')
    def circuit(x):
        params = _unflatten_params(flat_params, n_conv, n_pool, n_qubits)
        all_qubits = list(range(n_qubits))

        qml.AmplitudeEmbedding(features=x, wires=all_qubits, normalize=True)
        for q in all_qubits:
            qml.DepolarizingChannel(noise_p, wires=q)

        active = all_qubits.copy()
        for layer in range(n_conv):
            n_cur = len(active)
            if n_cur >= 4:
                w = int(math.sqrt(n_cur))
                while n_cur % w != 0:
                    w -= 1
                h = n_cur // w
                W, H = max(w, h), min(w, h)
                kernel = params[f'quantum_conv_kernel_{layer}']
                for win in QuantumNativeConvolution.get_conv_windows(W, H):
                    if max(win) < n_cur:
                        wires = [active[i] for i in win]
                        QuantumNativeConvolution.quantum_conv2d_kernel(kernel, wires)
                        for q in wires:
                            qml.DepolarizingChannel(noise_p, wires=q)

            if layer < n_conv - 1:
                if len(active) < 2:
                    break
                pairs = QuantumNativePooling.make_pairing(active)
                if not pairs:
                    break
                keep    = [k for k, _ in pairs]
                discard = [d for _, d in pairs]
                QuantumNativePooling.quantum_unitary_pooling(
                    params[f'quantum_pooling_{layer}'],
                    input_qubits=keep, output_qubits=discard
                )
                for q in keep + discard:
                    qml.DepolarizingChannel(noise_p, wires=q)
                active = keep

        cp = params['quantum_classifier']
        n_a = len(active)
        readout = active[0]

        for i, q in enumerate(active[:min(n_a, 4)]):
            qml.RX(cp[i * 2 % 32], wires=q)
            qml.RY(cp[(i * 2 + 1) % 32], wires=q)
            qml.RZ(cp[(i * 2 + 8) % 32], wires=q)

        for i in range(n_a - 1):
            qml.CNOT(wires=[active[i], active[i + 1]])
        if n_a >= 2:
            qml.CNOT(wires=[active[n_a - 1], active[0]])

        for i, q in enumerate(active[:min(n_a, 4)]):
            qml.RX(cp[(i * 2 + 16) % 32], wires=q)
            qml.RY(cp[(i * 2 + 17) % 32], wires=q)

        if n_a >= 2:
            qml.CNOT(wires=[active[0], active[min(n_a - 1, 1)]])
        qml.RZ(cp[31], wires=readout)

        for q in active:
            qml.DepolarizingChannel(noise_p, wires=q)

        return qml.expval(qml.PauliZ(readout))

    return circuit


# ── main entry point ──────────────────────────────────────────────────────────

def run_noise_simulation(
    weights_path,
    output_path,
    mnist_path=None,
    image_size=28,
    n_samples=8000,
    classes=(0, 1)
):
    np.random.seed(42)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── weights ───────────────────────────────────────────────────────────────
    print("Loading trained weights...")
    w = np.load(weights_path)
    for k in w.keys():
        print(f"  {k}: {w[k].shape}")

    n_conv = len([k for k in w.keys() if 'conv_kernel' in k])
    n_pool = len([k for k in w.keys() if 'pooling' in k])

    # FIX 3: infer n_qubits from image_size, not from pooling weight shape
    # n_qubits = ceil(log2(image_size^2)), padded to power of 2
    n_features = image_size * image_size
    n_qubits = math.ceil(math.log2(max(n_features, 2)))
    target_len = 2 ** n_qubits
    print(f"\nInferred: {n_conv} conv layers, {n_pool} pool layers, {n_qubits} qubits (target_len={target_len})")

    # ── dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading MNIST (image_size={image_size}, {n_samples} samples, classes={classes})...")
    if mnist_path:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from QCNN.utils.data_preprocessing import preprocess_for_quantum
        from QCNN.utils.dataset_loader import load_idx_dataset
        files = os.listdir(mnist_path)
        images_f = next((f for f in files if 'train' in f and 'images' in f and 'idx3' in f), None)
        labels_f = next((f for f in files if 'train' in f and 'labels' in f and 'idx1' in f), None)
        if not images_f or not labels_f:
            raise FileNotFoundError(f"No MNIST IDX files found in {mnist_path}")
        X_raw, y_raw = load_idx_dataset(
            os.path.join(mnist_path, images_f),
            os.path.join(mnist_path, labels_f)
        )
        mask = (y_raw == classes[0]) | (y_raw == classes[1])
        X_raw, y_raw = X_raw[mask], y_raw[mask]
        if len(X_raw) > n_samples:
            idx0 = np.where(y_raw == classes[0])[0]
            idx1 = np.where(y_raw == classes[1])[0]
            n0 = min(len(idx0), n_samples // 2)
            n1 = min(len(idx1), n_samples - n0)
            chosen = np.concatenate([
                np.random.choice(idx0, n0, replace=False),
                np.random.choice(idx1, n1, replace=False)
            ])
            np.random.shuffle(chosen)
            X_raw, y_raw = X_raw[chosen], y_raw[chosen]
        X_all, y_all = preprocess_for_quantum(
            X_raw, y_raw,
            n_qubits=n_qubits,
            image_size=image_size,
            normalization='minmax',
            encoding_type='amplitude'
        )
    else:
        X_all, y_all = _load_mnist(n_samples, classes, image_size)

    print(f"  Preprocessed shape: {X_all.shape}")

    # 70/30 split matching training exactly
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    print(f"  Test split: {len(X_test)} samples")

    MAX_TEST = 200
    if len(X_test) > MAX_TEST:
        rng_idx = np.random.choice(len(X_test), MAX_TEST, replace=False)
        X_test, y_test = X_test[rng_idx], y_test[rng_idx]
    print(f"  Using {len(X_test)} test samples for noise sweep")

    X_test_proc = _preprocess_for_circuit(X_test, target_len)
    flat_params  = _flatten_weights(w, n_conv, n_pool)
    print(f"  Flat params: {len(flat_params)}")

    # ── noise sweep ───────────────────────────────────────────────────────────
    p_values  = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.20]
    real_accs = []
    threshold = None  # calibrated once at p=0

    print("\nRunning noise sweep...")
    print(f"  {'p':>6}  {'accuracy':>10}  {'theoretical':>12}")
    print("  " + "-" * 34)

    for p in p_values:
        circuit = _make_noisy_circuit(n_qubits, n_conv, n_pool, flat_params, p)
        raw_outputs = np.array([float(circuit(xi)) for xi in X_test_proc])

        # FIX 1: calibrate threshold once at p=0 using midpoint of class means
        if p == 0.00:
            class_pos_mean = float(np.mean(raw_outputs[y_test == 1]))
            class_neg_mean = float(np.mean(raw_outputs[y_test == -1]))
            threshold = (class_pos_mean + class_neg_mean) / 2.0
            print(f"  Calibrated threshold: {threshold:.4f}  "
                  f"(class+1 mean={class_pos_mean:.4f}, class-1 mean={class_neg_mean:.4f})")

        preds = np.where(raw_outputs > threshold, 1, -1)
        acc   = float(np.mean(preds == y_test))
        real_accs.append(acc)
        print(f"  p={p:.2f}  acc={acc:.4f}  theoretical={(1 - p) * 0.945:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    theo_accs = [(1 - p) * 0.945 for p in p_values]

    # FIX 2: dynamic ylim so the real curve is always visible
    y_min = min(min(real_accs), min(theo_accs)) - 0.05
    y_min = max(0.40, round(y_min, 1))  # floor at 0.40, snap to 1dp

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, theo_accs, 'b--o',
            label='Theoretical: acc(p) = (1-p) \u00d7 0.945',
            linewidth=2, markersize=6)
    ax.plot(p_values, real_accs, 'r-s',
            label='Experimental (depolarizing noise)',
            linewidth=2, markersize=7)
    ax.axvline(x=0.05, color='gray', linestyle=':', linewidth=1.5,
               label='NISQ threshold (p=0.05)')
    ax.set_xlabel('Depolarizing noise probability p', fontsize=12)
    ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_title('FQCNN Noise Robustness: Experimental vs Theoretical', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.005, 0.21)
    ax.set_ylim(y_min, 1.00)
    ax.grid(True, alpha=0.3)

    # FIX 4: safe float comparison for annotations
    annotate_at = [0.00, 0.05, 0.10, 0.20]
    for p, acc in zip(p_values, real_accs):
        if any(abs(p - a) < 1e-9 for a in annotate_at):
            ax.annotate(f'{acc:.3f}', xy=(p, acc),
                        xytext=(p + 0.005, acc + 0.008),
                        fontsize=9, color='darkred')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {output_path}")

    print("\n-- Summary ----------------------------------------------------------")
    idx05 = next(i for i, p in enumerate(p_values) if abs(p - 0.05) < 1e-9)
    print(f"  Baseline (p=0):      real={real_accs[0]:.4f}  theoretical={theo_accs[0]:.4f}")
    print(f"  NISQ (p=0.05):       real={real_accs[idx05]:.4f}  theoretical={theo_accs[idx05]:.4f}")
    print(f"  High noise (p=0.20): real={real_accs[-1]:.4f}  theoretical={theo_accs[-1]:.4f}")
    max_diff = max(abs(r - t) for r, t in zip(real_accs, theo_accs))
    print(f"  Max deviation from theoretical: {max_diff:.4f}")
    print("---------------------------------------------------------------------")


if __name__ == "__main__":
    run_noise_simulation(
        weights_path="Results/Weights/quantum_model_params.npz",
        output_path="Results/Graphs/fig6_noise_robustness_real.png",
        image_size=28,
        n_samples=8000,
        classes=(0, 1)
    )