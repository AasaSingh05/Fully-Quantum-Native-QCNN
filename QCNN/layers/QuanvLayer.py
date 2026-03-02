import numpy as np
import pennylane as qml
import pennylane.numpy as pnp


class QuanvolutionalLayer:
    """
    Quanvolutional layer for processing images larger than the QCNN qubit count.

    Slides a parameterized quantum circuit over patches of the input image,
    measuring expectation values to produce a reduced quantum feature map.
    This is a quantum analog of a classical convolutional filter.

    Based on: Henderson et al., "Quanvolutional Neural Networks" (2019)
    and the PennyLane quanvolutional demo.
    """

    def __init__(self, patch_size: int = 4, n_filters: int = 4,
                 stride: int = 4, device_name: str = 'lightning.qubit',
                 random_params: bool = True, seed: int = 42):
        """
        Args:
            patch_size: Width/height of square patches (e.g. 4 → 16 qubits)
            n_filters: Number of independent quantum filters per patch
            stride: Step size between patches
            device_name: PennyLane device for the filter circuit
            random_params: If True, use fixed random parameters (no training)
            seed: Random seed for reproducible filter params
        """
        self.patch_size = patch_size
        self.n_qubits = patch_size ** 2
        self.n_filters = n_filters
        self.stride = stride
        self.device_name = device_name

        # Create device for the quanvolutional filter
        self.device = qml.device(device_name, wires=self.n_qubits)

        # Generate filter parameters
        rng = np.random.RandomState(seed)
        if random_params:
            # Fixed random filters (not trained — feature extraction only)
            # Each filter: n_qubits rotation angles + n_qubits entangling angles
            self.filter_params = [
                pnp.array(rng.uniform(0, 2 * np.pi, size=(self.n_qubits, 3)),
                          requires_grad=False)
                for _ in range(n_filters)
            ]
        else:
            # Trainable filters
            self.filter_params = [
                pnp.array(rng.normal(0, 0.1, size=(self.n_qubits, 3)),
                          requires_grad=True)
                for _ in range(n_filters)
            ]

        # Build the quantum filter circuit
        @qml.qnode(self.device, interface='autograd')
        def _quantum_filter(patch_data, filter_params):
            """Single quantum filter applied to a patch."""
            # Encode patch data via angle embedding
            for i in range(self.n_qubits):
                if i < len(patch_data):
                    qml.RY(patch_data[i], wires=i)

            # Parameterized variational layer
            for i in range(self.n_qubits):
                qml.RY(filter_params[i, 0], wires=i)
                qml.RZ(filter_params[i, 1], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Second rotation layer
            for i in range(self.n_qubits):
                qml.RY(filter_params[i, 2], wires=i)

            # Measure first qubit as filter output
            return qml.expval(qml.PauliZ(0))

        self._quantum_filter = _quantum_filter

    def extract_patches(self, image: np.ndarray) -> list:
        """
        Extract patches from a 2D image.

        Args:
            image: 2D array of shape (height, width)

        Returns:
            List of (flattened_patch, row_idx, col_idx) tuples
        """
        h, w = image.shape
        patches = []
        for r in range(0, h - self.patch_size + 1, self.stride):
            for c in range(0, w - self.patch_size + 1, self.stride):
                patch = image[r:r + self.patch_size, c:c + self.patch_size]
                patches.append((patch.flatten(), r, c))
        return patches

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply quanvolutional filters to an image and produce a reduced feature map.

        Args:
            image: 2D array (height, width) or 1D flattened array

        Returns:
            1D feature vector (out_h * out_w * n_filters,)
        """
        # Handle flattened input
        if image.ndim == 1:
            side = int(np.sqrt(len(image)))
            if side * side != len(image):
                # Pad to nearest square
                side = int(np.ceil(np.sqrt(len(image))))
                padded = np.zeros(side * side)
                padded[:len(image)] = image
                image = padded
            image = image.reshape(side, side)

        h, w = image.shape

        # Normalize to [0, 2π] if not already
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-10:
            image = (image - img_min) / (img_max - img_min) * 2 * np.pi

        patches = self.extract_patches(image)

        # Apply each filter to each patch
        out_h = (h - self.patch_size) // self.stride + 1
        out_w = (w - self.patch_size) // self.stride + 1

        feature_map = np.zeros((out_h, out_w, self.n_filters))

        for patch_data, r, c in patches:
            r_idx = r // self.stride
            c_idx = c // self.stride
            for f_idx, f_params in enumerate(self.filter_params):
                result = self._quantum_filter(patch_data, f_params)
                feature_map[r_idx, c_idx, f_idx] = float(result)

        return feature_map.flatten()

    def _build_thread_local_filter(self):
        """
        Create a thread-local QNode + device pair.
        PennyLane QNodes are NOT thread-safe (they share internal measurement
        state), so each worker thread must have its own device and QNode.
        """
        dev = qml.device(self.device_name, wires=self.n_qubits)
        n_qubits = self.n_qubits

        @qml.qnode(dev, interface='autograd')
        def _quantum_filter(patch_data, filter_params):
            for i in range(n_qubits):
                if i < len(patch_data):
                    qml.RY(patch_data[i], wires=i)
            for i in range(n_qubits):
                qml.RY(filter_params[i, 0], wires=i)
                qml.RZ(filter_params[i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(filter_params[i, 2], wires=i)
            return qml.expval(qml.PauliZ(0))

        return _quantum_filter

    def _process_image_with_filter(self, image: np.ndarray, qfilter) -> np.ndarray:
        """Process a single image using a given (thread-local) quantum filter."""
        if image.ndim == 1:
            side = int(np.sqrt(len(image)))
            if side * side != len(image):
                side = int(np.ceil(np.sqrt(len(image))))
                padded = np.zeros(side * side)
                padded[:len(image)] = image
                image = padded
            image = image.reshape(side, side)

        h, w = image.shape
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-10:
            image = (image - img_min) / (img_max - img_min) * 2 * np.pi

        patches = self.extract_patches(image)
        out_h = (h - self.patch_size) // self.stride + 1
        out_w = (w - self.patch_size) // self.stride + 1
        feature_map = np.zeros((out_h, out_w, self.n_filters))

        for patch_data, r, c in patches:
            r_idx = r // self.stride
            c_idx = c // self.stride
            for f_idx, f_params in enumerate(self.filter_params):
                result = qfilter(patch_data, f_params)
                feature_map[r_idx, c_idx, f_idx] = float(result)

        return feature_map.flatten()

    def process_batch(self, X: np.ndarray, image_size: int = None) -> np.ndarray:
        """
        Apply quanvolutional preprocessing to a batch of images using multiprocessing.
        Using separate processes avoids GIL issues and ensures better stability
        with PennyLane device initialization.

        Args:
            X: Batch of images, shape (n_samples, features) or (n_samples, h, w)
            image_size: If X is flat, reshape to (image_size, image_size)

        Returns:
            Reduced feature array (n_samples, reduced_features)
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        
        total = len(X)
        print(f"  Starting multiprocess Quanv processing ({total} samples)...", flush=True)
        
        # We need to pass necessary parameters to the worker since it's a separate process
        worker_args = [
            (i, x, self.patch_size, self.n_filters, self.stride, self.device_name, self.filter_params, image_size)
            for i, x in enumerate(X)
        ]

        # Use a fixed number of workers (e.g. 4) to keep intensity low
        # Total CPU usage will be roughly (max_workers / total_cores)
        max_workers = 4
        
        results = [None] * total
        processed_count = 0
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_idx = {executor.submit(_quanv_worker_task, arg): arg[0] for arg in worker_args}
            
            # Use as_completed to get results as they finish
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, features = future.result()
                    results[idx] = features
                    processed_count += 1
                    
                    if processed_count % 10 == 0 or processed_count == total:
                        print(f"  Quanv preprocessing: {processed_count}/{total} samples", flush=True)
                except Exception as e:
                    print(f"  [ERROR] Worker for sample {idx} failed: {e}", flush=True)

        return np.array(results)

    def get_output_size(self, image_size: int) -> int:
        """
        Compute the output feature count for a given image size.

        Args:
            image_size: Width/height of input image

        Returns:
            Total number of output features
        """
        out_dim = (image_size - self.patch_size) // self.stride + 1
        return out_dim * out_dim * self.n_filters

def _quanv_worker_task(args):
    """
    Independent worker task for multiprocessing.
    Must be a top-level function for pickling support.
    """
    import os
    # CRITICAL: Limit internal threading to 1 to avoid CPU oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    idx, img, patch_size, n_filters, stride, device_name, filter_params, image_size = args
    import pennylane as qml
    import numpy as np
    
    # Reshape if needed
    if img.ndim == 1 and image_size is not None:
        img = img.reshape(image_size, image_size)
    
    # 1. Initialize thread/process local device and QNode
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

    # 2. Process image
    h, w = img.shape
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-10:
        img = (img - img_min) / (img_max - img_min) * 2 * np.pi

    patches = []
    # Local worker cache for unique patches within THIS image
    # For MNIST, many patches are identical (all zeros)
    patch_cache = {}

    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = img[r:r + patch_size, c:c + patch_size]
            patches.append((patch.flatten(), r, c))

    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1
    feature_map = np.zeros((out_h, out_w, n_filters))

    for patch_data, r, c in patches:
        r_idx = r // stride
        c_idx = c // stride
        
        # Check if we've already processed this exact patch content
        patch_key = tuple(np.round(patch_data, 6))
        
        if patch_key in patch_cache:
            results = patch_cache[patch_key]
        else:
            results = []
            for f_idx, f_params in enumerate(filter_params):
                res = float(_quantum_filter(patch_data, f_params))
                results.append(res)
            patch_cache[patch_key] = results
            
        for f_idx, res in enumerate(results):
            feature_map[r_idx, c_idx, f_idx] = res

    return idx, feature_map.flatten()
