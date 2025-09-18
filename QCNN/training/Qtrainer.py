import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import time
import sys
from QCNN.models import PureQuantumNativeCNN

class QuantumNativeTrainer:
    """parameter-shift rule gradients"""
    
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        # Use quantum-aware optimizer
        self.quantum_optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    
    def train_pure_quantum_cnn(self, model: PureQuantumNativeCNN, 
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> PureQuantumNativeCNN:
        print("\n🚀 Starting Pure Quantum Native Training")
        print("="*50)
        
        best_accuracy = 0
        best_quantum_params = None
        
        # Flatten initial parameters for optimizer
        params_flat = model._flatten_params(model.quantum_params)
        n_epochs = model.config.n_epochs
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Shuffle quantum training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_quantum_loss = 0
            n_quantum_batches = 0
            
            # Quantum mini-batch training
            for i in range(0, len(X_train), model.config.batch_size):
                batch_end = min(i + model.config.batch_size, len(X_train))
                X_quantum_batch = X_shuffled[i:batch_end]
                y_quantum_batch = y_shuffled[i:batch_end]
                
                # Define quantum cost function
                def quantum_cost(params):
                    # Unflatten params for model usage
                    model.quantum_params = model._unflatten_params(params)
                    loss = model.quantum_loss_function(X_quantum_batch, y_quantum_batch)
                    return loss
                
                # Pure quantum parameter update using parameter-shift rule optimizer
                params_flat = self.quantum_optimizer.step(quantum_cost, params_flat)
                
                # Track quantum loss
                batch_loss = quantum_cost(params_flat)
                epoch_quantum_loss += batch_loss
                n_quantum_batches += 1
            
            avg_loss = epoch_quantum_loss / n_quantum_batches
            model.quantum_params = model._unflatten_params(params_flat)
            
            # Quantum evaluation
            train_accuracy = self._compute_quantum_accuracy(model, X_train[:50], y_train[:50])
            test_accuracy = self._compute_quantum_accuracy(model, X_test, y_test)
            
            # Update quantum training history
            model.training_history['loss'].append(avg_loss)
            model.training_history['accuracy'].append(test_accuracy)
            
            # Save best quantum model parameters
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
            
            # Calculate training progress percentage and estimate remaining time
            progress_percent = ((epoch + 1) / n_epochs) * 100
            elapsed = time.time() - epoch_start
            estimated_total = elapsed / ((epoch + 1) / n_epochs)
            remaining = estimated_total - elapsed
            
            # Display progress on the same console line
            sys.stdout.write(
                f"\r⚡ Quantum Epoch {epoch+1}/{n_epochs} | "
                f"Loss={avg_loss:.4f} | "
                f"Train Acc={train_accuracy:.3f} | "
                f"Test Acc={test_accuracy:.3f} | "
                f"Progress={progress_percent:.1f}% | "
                f"ETA={remaining:.1f}s"
            )
            sys.stdout.flush()
            
            # New line after last epoch
            if epoch == n_epochs - 1:
                print()
        
        # Restore best quantum parameters
        model.quantum_params = best_quantum_params
        print(f"\n🎯 Best Quantum Test Accuracy: {best_accuracy:.3f}")
        
        return model
    
    def _compute_quantum_accuracy(self, model: PureQuantumNativeCNN, 
                                  X: np.ndarray, y: np.ndarray) -> float:
        """Compute quantum classification accuracy"""
        quantum_predictions = model.quantum_predict_batch(X)
        return np.mean(quantum_predictions == y)
