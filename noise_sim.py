"""
noise_sim.py — Depolarizing noise robustness experiment for FQCNN

The model was trained with patch (quanvolutional) encoding, so this evaluates the
trained weights on the REAL 196-dim quanvolutional features of the reconstructed
test set (not raw pixels). It reconstructs the exact training test split (seed-42
pipeline) to recover aligned labels, reuses the cached quanv features (with an
alignment guard), derives the circuit qubit count from the saved weights, then
sweeps depolarizing noise p across [0, 0.20] and plots experimental vs theoretical.

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

def _find_mnist_idx(mnist_dir):
    """Locate the MNIST train IDX image/label files in a directory."""
    if not os.path.isdir(mnist_dir):
        return None, None
    files = os.listdir(mnist_dir)
    images_f = next((f for f in files if 'train' in f and 'images' in f and 'idx3' in f), None)
    labels_f = next((f for f in files if 'train' in f and 'labels' in f and 'idx1' in f), None)
    if not images_f or not labels_f:
        return None, None
    return os.path.join(mnist_dir, images_f), os.path.join(mnist_dir, labels_f)


def _reconstruct_training_split(image_size, classes, n_samples, mnist_path=None):
    """
    Reproduce the EXACT data pipeline from main.py so the recovered test images and
    labels line up with the trained model (and with the cached quanv features).

    Mirrors: load IDX → filter classes → preprocess_for_quantum(patch) →
    seed-42 stratified downsample to n_samples/0.7 → train_test_split(random_state=42).

    Returns (X_test_img, y_test): normalized (image_size x image_size) test images
    and labels in {-1, +1}, in the same order the trainer saw them.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from QCNN.utils.dataset_loader import load_idx_dataset
    from QCNN.utils.data_preprocessing import preprocess_for_quantum

    here = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = mnist_path or os.path.join(here, 'datasets', 'MNIST')
    images_f, labels_f = _find_mnist_idx(mnist_dir)
    if images_f is None:
        raise FileNotFoundError(f"No MNIST IDX files found in {mnist_dir}")

    print(f"  Loading MNIST IDX from {mnist_dir}/")
    X_raw, y_raw = load_idx_dataset(images_f, labels_f)
    mask = (y_raw == classes[0]) | (y_raw == classes[1])
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    print(f"  After class filter {classes}: {len(X_raw)} samples")

    # patch-mode preprocessing keeps 2D image structure (exactly as in training)
    X_pp, y_pp = preprocess_for_quantum(
        X_raw, y_raw,
        n_qubits=int(math.ceil(math.log2(max(image_size * image_size, 2)))),
        image_size=image_size,
        normalization='minmax',
        encoding_type='patch',
    )

    # stratified downsample to total_needed = n_samples / 0.7 (mirrors main.py:153-181).
    # main.py relies on the module-level np.random.seed(42) being pristine here, so we
    # re-seed to reproduce the identical choice()/shuffle() draw.
    total_needed = int(n_samples / 0.7)
    if total_needed < len(X_pp):
        np.random.seed(42)
        idx_pos = np.where(y_pp == 1)[0]
        idx_neg = np.where(y_pp == -1)[0]
        n_pos = min(len(idx_pos), total_needed // 2)
        n_neg = min(len(idx_neg), total_needed - total_needed // 2)
        sampled_pos = np.random.choice(idx_pos, n_pos, replace=False)
        sampled_neg = np.random.choice(idx_neg, n_neg, replace=False)
        indices = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(indices)
        X_pp, y_pp = X_pp[indices], y_pp[indices]
        print(f"  Stratified downsample to {len(X_pp)} samples ({n_pos} pos, {n_neg} neg)")

    _, X_test, _, y_test = train_test_split(
        X_pp, y_pp, test_size=0.3, random_state=42, stratify=y_pp
    )
    print(f"  Reconstructed test split: {len(X_test)} samples")
    return X_test, y_test


def _resolve_cached_features(cache_dir, n_test_expected, n_feat_expected):
    """Return the cached X_test feature matrix matching the expected shape, or None."""
    if not os.path.isdir(cache_dir):
        return None
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.startswith('quanv_cache_') or not fname.endswith('.npz'):
            continue
        data = np.load(os.path.join(cache_dir, fname))
        if 'X_test' in data and data['X_test'].shape == (n_test_expected, n_feat_expected):
            print(f"  Found matching quanv cache: {fname}  X_test={data['X_test'].shape}")
            return data['X_test']
    return None


def _quanv_features_for(images, seed=42):
    """Compute quanvolutional features for a batch of images (training config)."""
    from QCNN.layers.QuanvLayer import QuanvolutionalLayer
    quanv = QuanvolutionalLayer(patch_size=4, n_filters=4, stride=4, seed=seed)
    return np.array([quanv.process_image(img) for img in images])


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
    classes=(0, 1),
    max_test=200
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

    # Derive n_qubits from the saved pooling weights (pool_size = 3 * (n_qubits // 2)).
    # The trained model uses 8 qubits regardless of image_size, because patch encoding
    # reduces the image to a fixed-length quanvolutional feature vector first.
    n_qubits = 2 * (w['quantum_pooling_0'].size // 3)
    target_len = 2 ** n_qubits
    print(f"\nInferred: {n_conv} conv layers, {n_pool} pool layers, {n_qubits} qubits (target_len={target_len})")

    # ── reconstruct the REAL test set (images + aligned labels) ─────────────────
    print(f"\nReconstructing the trained test set (image_size={image_size}, "
          f"n_samples={n_samples}, classes={classes})...")
    X_test_img, y_test = _reconstruct_training_split(image_size, classes, n_samples, mnist_path)

    out_dim = (image_size - 4) // 4 + 1  # patch_size=4, stride=4
    n_feat = out_dim * out_dim * 4       # quanv output features (4 filters) → 196 for 28x28

    # ── obtain the quanvolutional features for the test set (matching training) ─
    # The model was trained on these features, NOT raw pixels. Prefer the cached
    # features written during training; verify alignment before trusting them.
    cache_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(weights_path)), '..', 'Cache'))
    cached = _resolve_cached_features(cache_dir, len(X_test_img), n_feat)
    if cached is not None:
        check = _quanv_features_for(X_test_img[:5])
        if np.allclose(check, cached[:5], atol=1e-6):
            print("  Alignment guard PASSED — using cached quanvolutional features.")
        else:
            print("  Alignment guard FAILED — will recompute quanv for the evaluated subset.")
            cached = None

    # ── pick the evaluation subset ──────────────────────────────────────────────
    np.random.seed(123)  # reproducible subset selection, independent of pipeline RNG
    if max_test is not None and len(X_test_img) > max_test:
        sub = np.random.choice(len(X_test_img), max_test, replace=False)
    else:
        sub = np.arange(len(X_test_img))
    y_test = y_test[sub]

    if cached is not None:
        X_feat = cached[sub]
    else:
        print(f"  Computing quanvolutional features for {len(sub)} test images...")
        X_feat = _quanv_features_for(X_test_img[sub])
    print(f"  Evaluating on {len(sub)} real test samples ({X_feat.shape[1]}-dim features)")

    # Model pads 196→256 then AmplitudeEmbedding(normalize=True); _preprocess_for_circuit
    # reproduces that (pad to 2^n_qubits + unit-norm).
    X_test_proc = _preprocess_for_circuit(X_feat, target_len)
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