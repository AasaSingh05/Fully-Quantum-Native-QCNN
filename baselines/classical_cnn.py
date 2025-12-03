# Classical CNN baseline for comparison with QCNN
# Simple 4x4 input model with minimal parameters

import numpy as np
import os
import matplotlib.pyplot as plt

# tensorflow import moved inside try to avoid import errors
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    raise ImportError("TensorFlow not installed. Install via: pip install tensorflow")

def build_baseline_cnn():
    """Builds a very small CNN model for 4x4 inputs.
       This model acts as a classical baseline to compare against QCNN."""
    
    model = models.Sequential()
    model.add(layers.Input(shape=(4,4,1)))
    model.add(layers.Conv2D(4, kernel_size=2, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='tanh'))     # match QCNN output range

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def train_baseline_cnn(X_train, y_train, X_test, y_test, epochs=40):
    """Trains classical CNN baseline and returns model + training history."""
    
    # reshape to (N, 4,4,1)
    X_train_cnn = X_train.reshape(-1,4,4,1)
    X_test_cnn = X_test.reshape(-1,4,4,1)

    model = build_baseline_cnn()
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=epochs,
        batch_size=16,
        verbose=0
    )
    return model, history


def save_baseline_plots(history, save_dir="Results/Baselines"):
    """Saves loss and accuracy plots for baseline CNN."""
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "classical_cnn_results.png")

    plt.figure(figsize=(12,4))

    # Training loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title("Classical CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy
    if 'accuracy' in history.history:
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Test Acc')
        plt.title("Classical CNN Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\n Baseline CNN plots saved to '{path}'")


def run_classical_baseline(X_train, y_train, X_test, y_test):
    """Convenience function to train baseline + save plots."""
    
    model, history = train_baseline_cnn(X_train, y_train, X_test, y_test)
    save_baseline_plots(history)

    # compute accuracy manually
    preds = np.where(model.predict(X_test.reshape(-1,4,4,1)) > 0, 1, -1)
    acc = np.mean(preds.flatten() == y_test)
    print(f"\n Classical CNN Test Accuracy: {acc:.1%}")
    return model, acc
