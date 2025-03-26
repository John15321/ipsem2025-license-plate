# Datasets Module

This module provides dataset management for the IPSEM 2025 License Plate Recognition project.

## Supported Datasets

### 1. EMNIST Dataset
- Balanced dataset optimized for license plate characters
- 36 classes (0-9, A-Z)
- Auto-downloading and caching
- 64x64 grayscale images
- Pre-configured for the hybrid model
- Automatic train/val/test splitting

### 2. MNIST Dataset
- Classic handwritten digit dataset
- 10 classes (0-9)
- Compatible with QNet architecture
- Built-in data augmentation

### 3. Custom Image Dataset
- User-provided image collections
- Flexible directory structure
- Multiple image formats
- Automatic preprocessing
- Memory-efficient loading

## CLI Usage

```bash
# Dataset operations
ipsem2025-dataset download --dataset-type emnist
ipsem2025-dataset info --dataset-path data
ipsem2025-dataset preview --dataset-path data --num-samples 8
ipsem2025-dataset validate --dataset-path custom_data
```

### CLI Options
- `--dataset-type`: emnist, mnist, or custom
- `--dataset-path`: Path to dataset
- `--num-samples`: Number of preview samples
- `--batch-size`: Batch size for operations
- `--verbose`: Enable debug output

## Python API

### Base Dataset
```python
from ipsem2025_license_plate.datasets import BaseDataset

# Core functionality for all datasets:
- create_data_loaders()
- get_class_mapping()
- get_image_dimensions()
- get_num_classes()
```

### Loading Datasets
```python
from ipsem2025_license_plate.datasets import EMNISTDataset, MNISTDataset, CustomImageDataset

# EMNIST for license plates
emnist = EMNISTDataset(root="data", download=True)
train_loader, val_loader, test_loader = emnist.create_data_loaders(
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15
)

# MNIST for digits
mnist = MNISTDataset(root="data", download=True)

# Custom dataset
custom = CustomImageDataset(root="custom_data")
```

## Integration with QNet
All datasets are compatible with the HybridModel:
- Proper image dimensions (64x64)
- Normalized pixel values
- Correct label mapping
- Memory-efficient batching
