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

    def __init__(self, learning_rate: float = 0.002, use_bce: bool = True):
        """
        Initialize trainer with learning rate, loss mode, and optimizer.

        Args:
            learning_rate: Adam stepsize for optimizing quantum parameters
            use_bce: whether to use BCE on mapped probabilities instead of MSE
        """
        self.learning_rate = learning_rate
        self.use_bce = use_bce
        # For quantum classification, 0.01 is generally a good start for BCE/MSE
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
        # Do not wrap logits_or_expvals in pnp.array() as it breaks the computational graph (ArrayBox)
        z = pnp.clip(logits_or_expvals, -1.0, 1.0)
        # Map ⟨Z⟩ in [-1,1] → p in [0,1]: +1 → p=1 (class 1), -1 → p=0 (class 0)
        # This is consistent with prediction threshold: ⟨Z⟩ > 0 → class +1
        p = (1.0 + z) * 0.5
        y01 = (pnp.array(labels_pm1) + 1.0) * 0.5
        eps = 1e-7
        
        # Use instance weights calculated in train_pure_quantum_cnn
        pos_w = getattr(self, '_pos_weight', 1.0)
        neg_w = getattr(self, '_neg_weight', 1.0)
        
        loss_pos = y01 * pnp.log(p + eps) * pos_w
        loss_neg = (1.0 - y01) * pnp.log(1.0 - p + eps) * neg_w
        
        return -pnp.mean(loss_pos + loss_neg)

    def train_pure_quantum_cnn(self, model: PureQuantumNativeCNN, 
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               log_filepath='quantum_training_log.txt',
                               summary_filepath='training_summary.txt',
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
        
        print(f"DEBUG: Qtrainer enters train_pure_quantum_cnn. Encoding: {model.config.encoding_type}")
        if model.config.encoding_type == 'patch' and model.quanv_layer is not None:
            import hashlib

            # Build a deterministic cache key from raw data shape + filter params
            cache_dir = os.path.join('Results', 'Cache')
            os.makedirs(cache_dir, exist_ok=True)

            hash_input = (
                f"{X_train.shape[1:]}_"  # Data resolution
                f"{model.quanv_layer.patch_size}_{model.quanv_layer.stride}_"
                f"{model.quanv_layer.n_filters}"
            )
            # Hash a signature of the combined data
            data_signature = np.ascontiguousarray(X_train[:100]).tobytes() + \
                            np.ascontiguousarray(X_test[:100]).tobytes()
            data_hash = hashlib.md5(data_signature).hexdigest()[:12]
            
            cache_key = hashlib.md5(
                (hash_input + data_hash).encode()
            ).hexdigest()[:16]
            cache_path = os.path.join(cache_dir, f"quanv_cache_{cache_key}.npz")

            if os.path.exists(cache_path):
                print(f"\n[CACHE HIT] Loading pre-calculated Quanvolutional Features from '{cache_path}'...")
                cached = np.load(cache_path)
                X_train = cached['X_train']
                X_test = cached['X_test']
                print(f"  Features loaded. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            else:
                n_train = len(X_train)
                total_samples = n_train + len(X_test)
                print(f"\n[CACHE MISS] Pre-calculating Quanvolutional Features for {total_samples} total samples...")
                print("  (This is an expensive one-time quantum simulation to speed up subsequent training)")
                
                # UNIFIED BATCHING: Combine for 100% core utilization
                X_combined = np.concatenate([X_train, X_test], axis=0)
                X_processed_all = model.quanv_layer.process_batch(X_combined)
                
                X_train = X_processed_all[:n_train]
                X_test = X_processed_all[n_train:]
                
                np.savez(cache_path, X_train=X_train, X_test=X_test)
                print(f"  Pre-calculation complete. Reduced shape: {X_train.shape[1:]}")
                print(f"  Cached to '{cache_path}' for future runs.")
            
            # Switch internal encoding to 'amplitude' for the QNode since data is now reduced
            model.config.encoding_type = 'amplitude'
        
        # Cache the (potentially preprocessed) datasets on the model so that
        # downstream evaluation can reuse the exact same representation that
        # was used during training and internal accuracy computation.
        model._quantum_preprocessed_train = X_train
        model._quantum_preprocessed_test = X_test
        
        # Calculate class weights for weighted BCE loss
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == -1)
        total = len(y_train)
        
        if pos_count > 0 and neg_count > 0:
            self._pos_weight = total / (2.0 * pos_count)
            self._neg_weight = total / (2.0 * neg_count)
            print(f"  Class Weights: +1 (Target): {self._pos_weight:.2f} | -1 (Rest): {self._neg_weight:.2f}")
        else:
            self._pos_weight = 1.0
            self._neg_weight = 1.0
        
        print("Starting Training")
        print("="*50)

        best_accuracy = 0
        best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
        
        # Flatten initial parameters for optimizer
        params_flat = model._flatten_params(model.quantum_params)
        ema_params_flat = params_flat.copy()
        ema_decay = getattr(model.config, 'ema_decay', 0.99)
        n_epochs = model.config.n_epochs
        
        # Stability parameters
        patience_counter = 0
        patience = getattr(model.config, 'early_stopping_patience', 3)
        lr_factor = getattr(model.config, 'lr_plateau_factor', 0.5)
        current_lr = model.config.learning_rate

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
                        X_processed = model._preprocess_input(X_quantum_batch)
                        preds = model.quantum_circuit(X_processed, params)
                        preds = pnp.atleast_1d(preds)
                        loss = self._bce_loss(preds, y_quantum_batch)
                    else:
                        loss = model.quantum_loss_function(X_quantum_batch, y_quantum_batch)
                    return loss
                
                # Manually compute gradients for clipping support
                grad, loss_val = self.quantum_optimizer.compute_grad(quantum_cost, (params_flat,), {})
                
                # Gradient clipping to prevent NaNs (softened for deeper model)
                grad_norm = pnp.linalg.norm(grad[0])
                if grad_norm > 5.0:
                    grad = (grad[0] * (5.0 / grad_norm),)
                
                params_flat = self.quantum_optimizer.apply_grad(grad, (params_flat,))[0]
                
                # Update Exponential Moving Average (EMA)
                ema_params_flat = (ema_decay * ema_params_flat) + ((1.0 - ema_decay) * params_flat)
                
                # Safe logging of real values
                epoch_quantum_loss += float(loss_val)
                n_quantum_batches += 1

                X_processed = model._preprocess_input(X_quantum_batch)
                
                # Diagnostic check (optional if speed is priority, but good for logs)
                if i == 0: # Only check first batch for stats to save time
                    quantum_outputs = pnp.atleast_1d(model.quantum_circuit(X_processed, params_flat))
                    print(
                        f"Epoch {epoch+1} Batch {i//model.config.batch_size+1} | "
                        f"Loss: {float(loss_val):.6f} | "
                        f"Stats: min: {quantum_outputs.min():.3f}, max: {quantum_outputs.max():.3f}, mean: {quantum_outputs.mean():.3f}"
                    )
                else:
                    print(f"Epoch {epoch+1} Batch {i//model.config.batch_size+1} | Loss: {float(loss_val):.6f}")
            
            avg_loss = epoch_quantum_loss / max(1, n_quantum_batches)
            
            # Switch to EMA parameters for evaluation
            model.quantum_params = model._unflatten_params(ema_params_flat)
            
            # Compute accuracies; limit train accuracy sample size for speed
            train_preds = model.quantum_predict_batch(X_train[:200])
            train_accuracy = np.mean(train_preds == y_train[:200])
            
            test_preds = model.quantum_predict_batch(X_test)
            test_accuracy = np.mean(test_preds == y_test)
            
            # Confusion matrix for diagnostics
            tp = np.sum((test_preds == 1) & (y_test == 1))
            tn = np.sum((test_preds == -1) & (y_test == -1))
            fp = np.sum((test_preds == 1) & (y_test == -1))
            fn = np.sum((test_preds == -1) & (y_test == 1))
            
            model.training_history['loss'].append(float(avg_loss))
            model.training_history['accuracy'].append(float(test_accuracy))
            
            elapsed = time.time() - epoch_start
            model.training_history['epoch_times'].append(float(elapsed))
            
            # Best Model Restoration & Stability Logic
            if test_accuracy > best_accuracy:
                print(f"  [STABILITY] Test accuracy improved: {best_accuracy:.2%} -> {test_accuracy:.2%} (via EMA). Saving best params.")
                best_accuracy = test_accuracy
                best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if current_lr > 1e-4:
                        current_lr *= lr_factor
                        self.quantum_optimizer.stepsize = current_lr
                        print(f"  [STABILITY] Plateau detected. Reducing Learning Rate to {current_lr:.6f}")
                        # Restore current params from best if we plateaued? 
                        # Usually we just drop LR and continue.
                        patience_counter = 0 # reset once per LR drop to allow recovery
                    else:
                        print(f"  [STABILITY] Early Stopping triggered after {epoch+1} epochs due to lack of improvement.")
                        break
            
            # Restore model to non-EMA params for the next epoch's training
            model.quantum_params = model._unflatten_params(params_flat)
            
            progress_percent = ((epoch + 1) / n_epochs) * 100
            estimated_total = elapsed / ((epoch + 1) / n_epochs)
            remaining = estimated_total - elapsed
            epoch_summary_text = (
                f"--- Epoch {epoch+1}/{n_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_accuracy:.1%} | "
                f"Test Acc: {test_accuracy:.1%} | "
                f"CM: [TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}] | "
                f"Progress: {progress_percent:.1f}% | "
                f"ETA: {remaining:.1f}s ---"
            )
            print(epoch_summary_text)

            # Write out to summary log
            if summary_filepath:
                try:
                    with open(summary_filepath, 'a', encoding='utf-8') as sf:
                        sf.write(epoch_summary_text + "\n")
                except Exception as e:
                    print(f"Warning: could not write summary to {summary_filepath}: {e}")
        
        # Restore best quantum parameters
        model.quantum_params = best_quantum_params
        
        final_msg = f"\nFinal Best Test Accuracy: {best_accuracy:.2%}\n"
        print(final_msg)
        if summary_filepath:
            with open(summary_filepath, 'a', encoding='utf-8') as sf:
                sf.write(final_msg)
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
            # In amplitude mode, X.shape[1] must be <= 2^n_qubits (QCNNModel handles padding)
            expected = 2 ** n_qubits
            if X.shape[1] > expected:
                raise ValueError(f"Amplitude encoding expects at most {expected} features for {n_qubits} qubits, got {X.shape[1]}")
        
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
                f"Set classes in Qconfig.py (e.g., classes = (0, 1)) to filter exactly two classes."
            )
