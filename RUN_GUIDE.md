# 📖 QCNN Dataset Running Guide

This project supports a wide range of formats. Follow this guide to run the training on your data.

## 📁 Standard Folder Structure
Place your datasets in the `datasets/` folder (created at the project root).
```text
datasets/
├── MNIST/              <-- Standard IDX-UBYTE files
├── images/             <-- Folders of png/jpg images
└── your_data.npz       <-- NumPy compressed data
```

## 🚀 Running with `runApp.sh`
The easiest way to run training is using the provided shell script.

### 1. Run your provided MNIST dataset (IDX format)
The script now defaults to your `datasets/MNIST` folder!
```bash
# Runs matching (0 vs 1) from datasets/MNIST
./runApp.sh
```

### 2. Run specific digits
```bash
# Run (3 vs 7) from the MNIST folder
./runApp.sh idx datasets/MNIST auto 28 3 7
```

### 3. Run Custom Images
Ensure your images are in `datasets/images/class_A` and `datasets/images/class_B`.
```bash
./runApp.sh images datasets/images
```

### 4. Run NPZ or CSV
```bash
./runApp.sh npz datasets/test_data.npz
```

---

## 🛠️ Advanced Usage (`main.py`)
You can use the Python script directly for full control:
```bash
python main.py --dataset idx --path datasets/MNIST --classes 0 1 --samples 100 --encoding patch
```

### Available Arguments:
- `--dataset`: `auto`, `idx`, `images`, `npz`, `csv`, `mnist`
- `--path`: Path to file or directory
- `--classes`: Two integers for binary classification
- `--samples`: Limit number of training samples
- `--encoding`: `feature_map`, `amplitude`, `patch`
- `--image-size`: Input dimensions (e.g. 28 for MNIST)
