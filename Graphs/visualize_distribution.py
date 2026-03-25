import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from QCNN.utils.dataset_loader import load_mnist_subset
from QCNN.utils.data_preprocessing import preprocess_for_quantum
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

classes = (0, 1)
n_samples = 1000
image_size = 4
n_qubits = 16
encoding_type = 'feature_map'

print(f"Loading raw MNIST (classes {classes[0]} and {classes[1]})...")
X_raw, y_raw = load_mnist_subset(n_samples=n_samples, classes=classes, flatten=True)

# Count raw labels
raw_counts = Counter(y_raw)
# Sort by digit
raw_labels, raw_freqs = zip(*sorted(raw_counts.items()))

print("Preprocessing...")
X_processed, y_processed = preprocess_for_quantum(
    X=X_raw, y=y_raw,
    n_qubits=n_qubits,
    image_size=image_size,
    normalization='minmax',
    encoding_type=encoding_type
)

# Count processed labels
proc_counts = Counter(y_processed)
proc_labels, proc_freqs = zip(*sorted(proc_counts.items()))

# Plot 1: Before Preprocessing
plt.figure(figsize=(6, 5))
bars_raw = plt.bar([str(l) for l in raw_labels], raw_freqs, color='skyblue')
plt.title('Label Distribution Before Preprocessing (0-9)')
plt.xlabel('Original Digit Labels')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight classes
for i, label in enumerate(raw_labels):
    if label in classes:
        bars_raw[i].set_color('salmon')

plt.tight_layout()
os.makedirs('Results/Preprocessing_Images', exist_ok=True)
save_path_raw = 'Results/Preprocessing_Images/class_distribution_raw.png'
plt.savefig(save_path_raw, dpi=150)
plt.close()
print(f"Saved raw distribution graph to {save_path_raw}")

# Plot 2: After Preprocessing
plt.figure(figsize=(6, 5))
bars_proc = plt.bar([str(l) for l in proc_labels], proc_freqs, color=['orange', 'salmon'])
plt.title('Label Distribution After Preprocessing')
plt.xlabel('Encoded Binary Labels')
plt.ylabel('Frequency')
plt.xticks(range(len(proc_labels)), [str(l) for l in proc_labels])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
save_path_proc = 'Results/Preprocessing_Images/class_distribution_processed.png'
plt.savefig(save_path_proc, dpi=150)
plt.close()
print(f"Saved processed distribution graph to {save_path_proc}")
