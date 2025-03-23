# Datasets Module

This module provides a flexible and extensible dataset management system for the IPSEM 2025 License Plate Recognition project. It implements various dataset types needed for training and evaluating license plate character recognition models.

## Key Components

### 1. BaseDataset (base.py)
The foundation of our dataset system, providing:
- Common interface for all dataset implementations
- Standard data loading and splitting functionality
- Type-safe abstract base class
- PyTorch Dataset compatibility

### 2. EMNISTDataset (emnist.py)
Specialized dataset for character recognition training:
- Uses EMNIST Balanced dataset
- Filters to 36 relevant classes (0-9, A-Z)
- Handles automatic downloading and caching
- Provides consistent 64x64 grayscale images
- Implements custom label mapping for license plates

### 3. CustomImageDataset (custom.py)
Flexible loader for user-provided image collections:
- Supports class-based directory structure
- Handles multiple image formats (PNG, JPG, BMP)
- Automatic class discovery and mapping
- Built-in image preprocessing
- Memory-efficient loading

## Usage Examples

### Loading EMNIST Dataset
```python
from ipsem2025_license_plate.datasets import EMNISTDataset

# Basic usage
dataset = EMNISTDataset(root="data", download=True)

# With custom transform
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
dataset = EMNISTDataset(root="data", transform=transform)
```

### Loading Custom Dataset
```python
from ipsem2025_license_plate.datasets import CustomImageDataset

# Basic usage with default settings
dataset = CustomImageDataset(root="path/to/image/folders")

# Create data loaders
train_loader, val_loader, test_loader = dataset.create_data_loaders(
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.2
)
```

## Directory Structure for Custom Datasets
```
dataset_root/
├── 0/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
├── 1/
│   ├── image1.png
│   └── ...
├── A/
│   └── ...
└── B/
    └── ...
```

## Features
- Automatic data splitting (train/validation/test)
- Configurable image preprocessing
- Memory-efficient data loading
- Comprehensive error handling
- Full PyTorch integration
- Type hints and documentation
- Extensive test coverage

## Adding New Dataset Types
1. Inherit from `BaseDataset`
2. Implement required abstract methods:
   - `get_image_dimensions()`
   - `get_num_classes()`
   - `get_class_mapping()`
   - `__len__()` and `__getitem__()`
   - `from_path()`
3. Add appropriate tests
4. Update documentation

## Testing
Each dataset implementation has comprehensive tests covering:
- Data loading and preprocessing
- Class mapping and label handling
- Error conditions and edge cases
- Memory efficiency
- PyTorch integration

Run tests with:
```bash
tox -e test
```