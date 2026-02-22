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

    def process_batch(self, X: np.ndarray, image_size: int = None) -> np.ndarray:
        """
        Apply quanvolutional preprocessing to a batch of images using multithreading.

        Args:
            X: Batch of images, shape (n_samples, features) or (n_samples, h, w)
            image_size: If X is flat, reshape to (image_size, image_size)

        Returns:
            Reduced feature array (n_samples, reduced_features)
        """
        import concurrent.futures

        def process_single(args):
            i, x = args
            if x.ndim == 1 and image_size is not None:
                x = x.reshape(image_size, image_size)
            return i, self.process_image(x)

        results = [None] * len(X)
        
        # lightning.qubit operates in C++ and releases the GIL, making ThreadPool scalable
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single, (i, x)) for i, x in enumerate(X)]
            
            for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                i, features = future.result()
                results[i] = features
                
                if completed_count % 50 == 0 or completed_count == len(X):
                    print(f"  Quanv preprocessing: {completed_count}/{len(X)} samples")

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
