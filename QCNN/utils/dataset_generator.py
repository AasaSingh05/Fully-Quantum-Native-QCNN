import numpy as np
from typing import Tuple

def generate_quantum_binary_dataset(n_samples: int = 100, 
                                   image_size: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset optimized for quantum advantage
    Patterns that leverage quantum entanglement and superposition
    """
    np.random.seed(42)
    
    X_quantum = []
    y_quantum = []
    
    for _ in range(n_samples):
        # Create quantum-favorable patterns
        img = np.random.random((image_size, image_size)) * 0.1
        
        if np.random.random() > 0.5:
            # Pattern 1: Quantum superposition-like pattern (class +1)
            # High amplitude on diagonal, low elsewhere  
            for i in range(image_size):
                img[i, i] += 0.8
                img[i, (i + 1) % image_size] += 0.3  # Circular pattern
            label = 1
        else:
            # Pattern 2: Anti-diagonal quantum pattern (class -1)
            for i in range(image_size):
                img[i, image_size - 1 - i] += 0.8
                img[(i + 1) % image_size, image_size - 1 - i] += 0.3
            label = -1
        
        # Normalize for quantum encoding
        img_flat = img.flatten()
        img_normalized = (img_flat - np.mean(img_flat)) / (np.std(img_flat) + 1e-8)
        
        # Scale to appropriate range for quantum encoding
        img_scaled = (img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized) + 1e-8)
        
        X_quantum.append(img_scaled)
        y_quantum.append(label)
    
    return np.array(X_quantum), np.array(y_quantum)