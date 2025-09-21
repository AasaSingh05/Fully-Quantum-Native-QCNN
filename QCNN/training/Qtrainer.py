import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import time
import sys
import os
from QCNN.models import PureQuantumNativeCNN


class QuantumNativeTrainer:
    """parameter-shift rule gradients"""
    
    def __init__(self, learning_rate: float = 0.005):
        self.learning_rate = learning_rate
        # Use quantum-aware optimizer
        self.quantum_optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    
    def save_params(self, params: dict, filepath: str):
        """Save quantum circuit parameters to disk"""
        numpy_params = {k: np.array(v) for k, v in params.items()}
        np.savez(filepath, **numpy_params)
    
    def load_params(self, filepath: str):
        """Load quantum circuit parameters from disk"""
        data = np.load(filepath)
        return {k: pnp.array(data[k], requires_grad=True) for k in data.files}
    
    def train_pure_quantum_cnn(self, model: PureQuantumNativeCNN, 
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               log_filepath='quantum_training_log.txt') -> PureQuantumNativeCNN:
        with open(log_filepath, 'w') as log_file:
            log_file.write("Starting Pure Quantum Native Training\n")
            log_file.write("="*50 + "\n")

            best_accuracy = 0
            best_quantum_params = None
            
            # Flatten initial parameters for optimizer
            params_flat = model._flatten_params(model.quantum_params)
            n_epochs = model.config.n_epochs
            
            for epoch in range(n_epochs):
                epoch_start = time.time()
                
                indices = np.random.permutation(len(X_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                epoch_quantum_loss = 0
                n_quantum_batches = 0
                
                for i in range(0, len(X_train), model.config.batch_size):
                    batch_end = min(i + model.config.batch_size, len(X_train))
                    X_quantum_batch = X_shuffled[i:batch_end]
                    y_quantum_batch = y_shuffled[i:batch_end]
                    
                    def quantum_cost(params):
                        model.quantum_params = model._unflatten_params(params)
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
                    
                    # Log diagnostic info to file
                    log_file.write(
                        f"\nEpoch {epoch+1} Batch {i//model.config.batch_size+1} quantum output stats:\n"
                        f"  min: {quantum_outputs.min()}, max: {quantum_outputs.max()}, "
                        f"mean: {quantum_outputs.mean()}, std: {quantum_outputs.std()}\n"
                        f"Batch param norm before step: {param_norm_before}\n"
                        f"Batch param norm after step: {param_norm_after}\n"
                    )
                
                avg_loss = epoch_quantum_loss / n_quantum_batches
                model.quantum_params = model._unflatten_params(params_flat)
                
                train_accuracy = self._compute_quantum_accuracy(model, X_train[:50], y_train[:50])
                test_accuracy = self._compute_quantum_accuracy(model, X_test, y_test)
                
                model.training_history['loss'].append(avg_loss)
                model.training_history['accuracy'].append(test_accuracy)
                
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
                
                progress_percent = ((epoch + 1) / n_epochs) * 100
                elapsed = time.time() - epoch_start
                estimated_total = elapsed / ((epoch + 1) / n_epochs)
                remaining = estimated_total - elapsed
                
                log_file.write(
                    f"Epoch {epoch+1}/{n_epochs} | Loss={avg_loss:.4f} | "
                    f"Train Acc={train_accuracy:.3f} | Test Acc={test_accuracy:.3f} | "
                    f"Progress={progress_percent:.1f}% | ETA={remaining:.1f}s\n"
                )
        
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
        """Compute quantum classification accuracy"""
        quantum_predictions = model.quantum_predict_batch(X)
        return np.mean(quantum_predictions == y)
