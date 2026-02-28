import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor

def _quanv_worker_task(args):
    idx, img, patch_size, n_filters, stride, device_name, filter_params, image_size = args
    import pennylane as qml
    import numpy as np
    
    print(f"Worker {idx} starting...", flush=True)
    n_qubits = patch_size ** 2
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def _quantum_filter(patch_data, f_params):
        for i in range(n_qubits):
            if i < len(patch_data):
                qml.RY(patch_data[i], wires=i)
        for i in range(n_qubits):
            qml.RY(f_params[i, 0], wires=i)
            qml.RZ(f_params[i, 1], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits):
            qml.RY(f_params[i, 2], wires=i)
        return qml.expval(qml.PauliZ(0))

    h, w = img.shape
    patches = []
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = img[r:r + patch_size, c:c + patch_size]
            patches.append((patch.flatten(), r, c))

    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1
    feature_map = np.zeros((out_h, out_w, n_filters))

    for p_idx, (patch_data, r, c) in enumerate(patches):
        r_idx = r // stride
        c_idx = c // stride
        for f_idx, f_params in enumerate(filter_params):
            result = _quantum_filter(patch_data, f_params)
            feature_map[r_idx, c_idx, f_idx] = float(result)
        if (p_idx + 1) % 10 == 0:
            print(f"Worker {idx} progress: {p_idx+1}/{len(patches)} patches", flush=True)

    print(f"Worker {idx} finished.", flush=True)
    return idx, feature_map.flatten()

if __name__ == "__main__":
    patch_size = 4
    n_filters = 4
    stride = 4
    device_name = "lightning.qubit"
    n_samples = 4
    image_size = 28
    
    img = np.random.rand(image_size, image_size)
    filter_params = [np.random.rand(patch_size**2, 3) for _ in range(n_filters)]
    
    worker_args = [
        (i, img, patch_size, n_filters, stride, device_name, filter_params, image_size)
        for i in range(n_samples)
    ]
    
    print(f"Starting test with {n_samples} samples and {os.cpu_count()} CPUs...", flush=True)
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_quanv_worker_task, worker_args))
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s", flush=True)
    print(f"Results obtained: {len(results)}", flush=True)
