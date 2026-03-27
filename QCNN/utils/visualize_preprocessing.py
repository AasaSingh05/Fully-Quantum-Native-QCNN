import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
# This script is in QCNN/utils, so project root is 2 levels up
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import necessary functions
from QCNN.utils.dataset_loader import load_mnist_subset
from QCNN.utils.data_preprocessing import preprocess_for_quantum

def main():
    """
    Visualize MNIST samples before and after quantum preprocessing.
    The 'After' images are resized and normalized for quantum feature maps.
    """
    print("Loading raw MNIST samples from local IDX files (Classes 0 and 1)...")
    try:
        # Define paths to local IDX files
        mnist_dir = os.path.join(project_root, "datasets", "MNIST")
        images_path = os.path.join(mnist_dir, "t10k-images.idx3-ubyte")
        labels_path = os.path.join(mnist_dir, "t10k-labels.idx1-ubyte")
        
        from QCNN.utils.dataset_loader import load_idx_dataset
        X_all, y_all = load_idx_dataset(images_path, labels_path)
        
        # Filter for classes 0 and 1
        mask = (y_all == 0) | (y_all == 1)
        X_filtered = X_all[mask]
        y_filtered = y_all[mask]
        
        # Take first 4 samples
        X_raw = X_filtered[:4].reshape(-1, 28, 28)
        y_raw = y_filtered[:4]
    except Exception as e:
        print(f"Error loading local IDX files: {e}")
        print("Falling back to synthetic data...")
        X_raw = np.random.rand(4, 28, 28)
        y_raw = np.array([0, 1, 0, 1])

    print("Applying quantum preprocessing (32x32 target resolution)...")
    # Preprocess (X_raw is 4, 28, 28)
    # Using image_size=32 and encoding_type='patch' to maintain 2D structure
    X_proc, y_proc = preprocess_for_quantum(
        X_raw, y_raw, 
        n_qubits=8,         # Matches metadata.json
        image_size=32,      # Matches metadata.json
        normalization='minmax', 
        encoding_type='patch' # Keeps 2D for visualization
    )
    
    # X_proc is already (4, 32, 32) because encoding_type is 'patch'
    X_proc_reshaped = X_proc
    
    # Ensure output directory exists
    images_dir = os.path.join(project_root, "Results", "preprocessing_images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Generating 4 individual visualization plots in {images_dir}...")
    
    # Labels for better clarity
    before_cmap = 'gray'
    after_cmap = 'plasma'
    
    for i in range(4):
        # Create a 1x2 plot for each sample
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f"Sample {i+1} Preprocessing (Class {y_raw[i]} -> Label {y_proc[i]})", fontsize=14)
        
        # Before (Raw 28x28 image)
        axes[0].imshow(X_raw[i], cmap=before_cmap)
        axes[0].set_title(f"Before (28x28 Raw)", fontsize=11)
        axes[0].axis('off')
        
        # After (Preprocessed 32x32 image)
        im = axes[1].imshow(X_proc_reshaped[i], cmap=after_cmap)
        axes[1].set_title(f"After (32x32 Quantum-Native)", fontsize=11)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save individual sample image
        output_path = os.path.join(images_dir, f"sample_{i+1}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: sample_{i+1}.png")
    
    # Also keep the summary plot for convenience
    summary_path = os.path.join(project_root, "Results", "preprocessing_visualization.png")
    # ... (rest of the code for summary plot remains or is replaced)
    
    print("-" * 50)
    print(f"SUCCESS: Individual images saved to:\n  {images_dir}")
    print("-" * 50)

if __name__ == "__main__":
    main()
