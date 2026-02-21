import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import time
import sys
import os
from QCNN.models import PureQuantumNativeCNN


class QuantumNativeTrainer:
    """
    Trainer implementing parameter-shift-based optimization for QCNN models.
    Provides MSE or BCE objectives, Adam optimizer integration, checkpointing,
    and training diagnostics logged to file.
    """

    def __init__(self, learning_rate: float = 0.005, use_bce: bool = False):
        """
        Initialize trainer with learning rate, loss mode, and optimizer.

        Args:
            learning_rate: Adam stepsize for optimizing quantum parameters
            use_bce: whether to use BCE on mapped probabilities instead of MSE
        """
        self.learning_rate = learning_rate
        self.use_bce = use_bce  # optional BCE on mapped probability
        # Use quantum-aware optimizer
        self.quantum_optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    
    def save_params(self, params: dict, filepath: str):
        """
        Save quantum parameters to disk as .npz.

        Args:
            params: dict of trainable parameter arrays
            filepath: output file path
        """
        numpy_params = {k: np.array(v) for k, v in params.items()}
        np.savez(filepath, **numpy_params)
    
    def load_params(self, filepath: str):
        """
        Load saved quantum parameter tensors from .npz file.

        Args:
            filepath: path to saved weights

        Returns:
            dict of parameter names mapped to PennyLane arrays
        """
        data = np.load(filepath)
        return {k: pnp.array(data[k], requires_grad=True) for k in data.files}
    
    def _bce_loss(self, logits_or_expvals, labels_pm1):
        """
        Binary cross-entropy loss on probabilities p = (1 - <Z>)/2.
        Input labels are {-1, +1}; internally converted to {0,1}.

        Args:
            logits_or_expvals: model outputs in [-1,1]
            labels_pm1: labels in {-1,1}

        Returns:
            scalar BCE loss
        """
        z = pnp.clip(pnp.array(logits_or_expvals), -1.0, 1.0)
        p = (1.0 - z) * 0.5  # map <Z> in [-1,1] -> p in [0,1]
        y01 = (pnp.array(labels_pm1) + 1.0) * 0.5
        eps = 1e-8
        return -pnp.mean(y01 * pnp.log(p + eps) + (1.0 - y01) * pnp.log(1.0 - p + eps))

    def train_pure_quantum_cnn(self, model: PureQuantumNativeCNN, 
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               log_filepath='quantum_training_log.txt',
                               validate_data: bool = True) -> PureQuantumNativeCNN:
        """
        Train QCNN model end-to-end using parameter-shift-compatible optimization.
        Logs diagnostics per batch, tracks best test accuracy, applies checkpointing,
        and returns the trained model.

        Args:
            model: PureQuantumNativeCNN instance to optimize
            X_train, y_train: training dataset (preprocessed for quantum encoding)
            X_test, y_test: evaluation dataset (preprocessed for quantum encoding)
            log_filepath: where to write training diagnostics
            validate_data: whether to validate dataset compatibility

        Returns:
            model with best-performing parameters restored
        """
        # Validate dataset compatibility with quantum circuit
        if validate_data:
            self._validate_dataset(X_train, y_train, model)
            self._validate_dataset(X_test, y_test, model)
        
        print("Starting Training")
        print("="*50)

        best_accuracy = 0
        best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
        
        # Flatten initial parameters for optimizer
        params_flat = model._flatten_params(model.quantum_params)
        n_epochs = model.config.n_epochs
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_quantum_loss = 0.0
            n_quantum_batches = 0
            
            for i in range(0, len(X_train), model.config.batch_size):
                batch_end = min(i + model.config.batch_size, len(X_train))
                X_quantum_batch = X_shuffled[i:batch_end]
                y_quantum_batch = y_shuffled[i:batch_end]
                
                def quantum_cost(params):
                    model.quantum_params = model._unflatten_params(params)
                    if self.use_bce:
                        # Compute BCE on mapped probabilities
                        preds = [model.quantum_predict_single(x) for x in X_quantum_batch]
                        loss = self._bce_loss(preds, y_quantum_batch)
                    else:
                        # Original MSE on expectation values vs {-1,1} labels
                        loss = model.quantum_loss_function(X_quantum_batch, y_quantum_batch)
                    return loss
                
                param_norm_before = pnp.linalg.norm(params_flat)
                params_flat = self.quantum_optimizer.step(quantum_cost, params_flat)
                param_norm_after = pnp.linalg.norm(params_flat)

                batch_loss = quantum_cost(params_flat)
                epoch_quantum_loss += batch_loss
                n_quantum_batches += 1

                quantum_outputs = [model.quantum_predict_single(x) for x in X_quantum_batch]
                quantum_outputs = np.array(quantum_outputs)
                
                # Print diagnostic info (captured by Logger)
                print(
                    f"\nEpoch {epoch+1} Batch {i//model.config.batch_size+1} stats:"
                    f" min: {quantum_outputs.min():.3f}, max: {quantum_outputs.max():.3f}, "
                    f"mean: {quantum_outputs.mean():.3f}"
                )
            
            avg_loss = epoch_quantum_loss / max(1, n_quantum_batches)
            model.quantum_params = model._unflatten_params(params_flat)
            
            # Compute accuracies; limit train accuracy sample size for speed
            train_accuracy = self._compute_quantum_accuracy(model, X_train[:50], y_train[:50])
            test_accuracy = self._compute_quantum_accuracy(model, X_test, y_test)
            
            model.training_history['loss'].append(float(avg_loss))
            model.training_history['accuracy'].append(float(test_accuracy))
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
            
            progress_percent = ((epoch + 1) / n_epochs) * 100
            elapsed = time.time() - epoch_start
            estimated_total = elapsed / ((epoch + 1) / n_epochs)
            remaining = estimated_total - elapsed
            
            print(
                f"\n--- Epoch {epoch+1}/{n_epochs} Summary ---"
            )
            print(
                f"  Loss: {avg_loss:.4f}"
            )
            print(
                f"  Train Accuracy: {train_accuracy:.1%}"
            )
            print(
                f"  Test Accuracy: {test_accuracy:.1%}"
            )
            print(
                f"  Progress: {progress_percent:.1f}%"
            )
            print(
                f"  ETA: {remaining:.1f}s"
            )
            print("-" * 30)
        
        # Restore best quantum parameters
        model.quantum_params = best_quantum_params
        
        # Ensure Results/Weights directory exists
        weights_dir = os.path.join('Results', 'Weights')
        os.makedirs(weights_dir, exist_ok=True)

        weights_path = os.path.join(weights_dir, 'quantum_model_params.npz')
        self.save_params(model.quantum_params, weights_path)
        print(f"Saved trained quantum model parameters to '{weights_path}'")
        
        print(f"\nBest Quantum Test Accuracy: {best_accuracy:.3f}")
        
        return model
    
    def _compute_quantum_accuracy(self, model: PureQuantumNativeCNN, 
                                  X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary classification accuracy using quantum predictions.

        Args:
            model: trained QCNN
            X: samples
            y: target labels in {-1,1}

        Returns:
            scalar accuracy between 0 and 1
        """
        quantum_predictions = model.quantum_predict_batch(X)
        return np.mean(quantum_predictions == y)
    
    def _validate_dataset(self, X: np.ndarray, y: np.ndarray, model: 'PureQuantumNativeCNN'):
        """
        Validate dataset compatibility with quantum circuit requirements.

        Args:
            X: Feature array
            y: Label array
            model: The QCNN model instance

        Raises:
            ValueError if validation fails
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
        
        encoding = model.config.encoding_type
        n_qubits = model.config.n_qubits
        
        if encoding == 'patch':
            # In patch mode, X should be 3D (n_samples, h, w)
            if X.ndim != 3:
                raise ValueError(f"Patch encoding expects 3D array (n_samples, h, w), got {X.shape}")
            if X.shape[1] != model.config.image_size or X.shape[2] != model.config.image_size:
                print(f"Warning: Image size {X.shape[1:]} doesn't match config {model.config.image_size}")
        
        elif encoding == 'amplitude':
            # In amplitude mode, X.shape[1] should be 2^n_qubits
            expected = 2 ** n_qubits
            if X.shape[1] != expected:
                raise ValueError(f"Amplitude encoding expects {expected} features for {n_qubits} qubits, got {X.shape[1]}")
        
        else:
            # feature_map: 1 qubit per feature
            if X.shape[1] != n_qubits:
                raise ValueError(
                    f"Feature count {X.shape[1]} doesn't match n_qubits {n_qubits}. "
                    f"Please preprocess your data correctly for 'feature_map' encoding."
                )
        
        if not np.all((X >= -0.1) & (X <= 2 * np.pi + 0.1)):
            print(f"Warning: Features should be in [0, 2π] range for quantum encoding. "
                  f"Got [{X.min():.3f}, {X.max():.3f}]")
        
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([-1, 1])):
            raise ValueError(
                f"Labels must be {{-1, +1}} for binary classification. Got {unique_labels}. "
                f"Use data_preprocessing.encode_labels() to convert your labels."
            )
