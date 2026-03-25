import numpy as np
import matplotlib.pyplot as plt
import os
from QCNN.utils.dataset_loader import load_mnist_subset
from QCNN.utils.data_preprocessing import preprocess_for_quantum

# Configuration
classes = (0, 1)
image_size = 4
n_qubits = 16
encoding_type = 'feature_map'

print("Loading raw MNIST...")
# 1. Load raw MNIST (before preprocessing)
X_raw, y_raw = load_mnist_subset(n_samples=10, classes=classes, flatten=False)

# Get one sample from each class
idx_pos = np.where(y_raw == classes[1])[0][0]
idx_neg = np.where(y_raw == classes[0])[0][0]

X_pos_raw = X_raw[idx_pos]
X_neg_raw = X_raw[idx_neg]

print("Preprocessing...")
# 2. Preprocess
# load_mnist_subset returns float64, let's flatten it for the preprocessor
X_flat = X_raw.reshape(X_raw.shape[0], -1)

# During preprocessing: Image gets resized/cropped and normalized
# Let's run it through the actual pipeline
X_processed, y_processed = preprocess_for_quantum(
    X=X_flat, y=y_raw,
    n_qubits=n_qubits,
    image_size=image_size,
    normalization='minmax',
    encoding_type=encoding_type
)

X_pos_processed = X_processed[idx_pos].reshape(image_size, image_size)
X_neg_processed = X_processed[idx_neg].reshape(image_size, image_size)

# The preprocessed features for quantum encoding are in [0, 2pi].
# To visualize them properly in grayscale, we need to normalize them back to [0, 1]
# otherwise matplotlib might render values > 1 as clipped (often white or black depending on cmap)
X_pos_processed_vis = X_pos_processed / (2 * np.pi)
X_neg_processed_vis = X_neg_processed / (2 * np.pi)

os.makedirs('Results/Preprocessing_Images', exist_ok=True)

def save_image(img, title, filename):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

save_image(X_pos_raw, f'Raw: Class {classes[1]} (+1)', 'Results/Preprocessing_Images/raw_target.png')
save_image(X_neg_raw, f'Raw: Class {classes[0]} (-1)', 'Results/Preprocessing_Images/raw_nontarget.png')

save_image(X_pos_processed_vis, f'Processed: Label {y_processed[idx_pos]}', 'Results/Preprocessing_Images/processed_target.png')
save_image(X_neg_processed_vis, f'Processed: Label {y_processed[idx_neg]}', 'Results/Preprocessing_Images/processed_nontarget.png')

print("All images generated successfully.")
