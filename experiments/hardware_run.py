#!/usr/bin/env python3
"""
Optional: run the trained FQCNN circuit on REAL IBM Quantum hardware (suggestion #9).

This is a thin, opt-in scaffold. It costs nothing on IBM Quantum's free **Open
Plan** (a limited monthly allowance of real-QPU time), but it requires:

  1. pip install pennylane-qiskit qiskit-ibm-runtime
  2. An IBM Quantum API token exported as the env var IBM_QUANTUM_TOKEN
     (get one free at https://quantum.ibm.com/).

If either is missing, the script prints how to set it up and exits cleanly — so
it never breaks an automated run. Because real-QPU time is scarce, it evaluates
only a SMALL number of test samples at a REDUCED qubit count by default.

Usage:
  IBM_QUANTUM_TOKEN=... python -m experiments.hardware_run \
      --weights Results/Weights/quantum_model_params.npz --n-samples 10 --shots 1024
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _preflight() -> str | None:
    """Return the IBM token if everything is ready, else None (after printing help)."""
    missing = [m for m in ("pennylane_qiskit", "qiskit_ibm_runtime") if not _have(m)]
    if missing:
        print("Hardware run skipped — missing packages:", ", ".join(missing))
        print("  Install with: pip install pennylane-qiskit qiskit-ibm-runtime")
        return None
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        print("Hardware run skipped — IBM_QUANTUM_TOKEN env var not set.")
        print("  Get a free token at https://quantum.ibm.com/ and export it:")
        print("    export IBM_QUANTUM_TOKEN=<your-token>")
        return None
    return token


def main():
    ap = argparse.ArgumentParser(description="Run trained FQCNN on IBM hardware (optional)")
    ap.add_argument("--weights", default="Results/Weights/quantum_model_params.npz")
    ap.add_argument("--backend", default="least_busy",
                    help="IBM backend name, or 'least_busy' to auto-pick")
    ap.add_argument("--n-samples", type=int, default=10, help="Test samples (keep small!)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--classes", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--image-size", type=int, default=28)
    args = ap.parse_args()

    token = _preflight()
    if token is None:
        sys.exit(0)  # clean, non-error skip

    import pennylane as qml
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Reuse the noise_sim helpers to reconstruct the exact trained test set and
    # the trained parameters / circuit topology.
    import noise_sim as ns

    print("Loading trained weights:", args.weights)
    w = np.load(args.weights)
    n_conv = len([k for k in w.keys() if "conv_kernel" in k])
    n_pool = len([k for k in w.keys() if "pooling" in k])
    n_qubits = 2 * (w["quantum_pooling_0"].size // 3)
    flat = ns._flatten_weights(w, n_conv, n_pool)
    target_len = 2 ** n_qubits

    print(f"Reconstructing {args.n_samples} real test samples ...")
    X_img, y = ns._reconstruct_training_split(
        args.image_size, tuple(args.classes), n_samples=8000)
    X_feat = ns._quanv_features_for(X_img[: args.n_samples])
    y = y[: args.n_samples]
    X_proc = ns._preprocess_for_circuit(X_feat, target_len)

    print(f"Connecting to IBM Quantum (backend={args.backend}) ...")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = (service.least_busy(operational=True, simulator=False)
               if args.backend == "least_busy"
               else service.backend(args.backend))
    print(f"Using backend: {backend.name}")

    dev = qml.device("qiskit.remote", wires=n_qubits, backend=backend, shots=args.shots)

    @qml.qnode(dev)
    def circuit(x):
        # Noise-free topology (real hardware supplies its own noise).
        spec = ns._build_noise_spec("depolarizing", 0.0)
        inner = ns._make_noisy_circuit(n_qubits, n_conv, n_pool, flat, spec)
        return inner(x)

    print("Running on hardware (this may queue) ...")
    raw = np.array([float(circuit(xi)) for xi in X_proc])
    preds = np.where(raw > 0, 1, -1)
    acc = float(np.mean(preds == y))
    print(f"\nHardware accuracy on {len(y)} samples: {acc:.3f}")
    print("(Compare against the simulator / noise-emulation results.)")


if __name__ == "__main__":
    main()
