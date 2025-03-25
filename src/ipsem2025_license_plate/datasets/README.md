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

## Dataset CLI Tool

The dataset module includes a comprehensive command-line interface for managing and working with datasets.

### Installation

The CLI is installed automatically with the package and available as `ipsem2025-dataset`:

```bash
# Install the package
pip install ipsem2025-license-plate

# Verify installation
ipsem2025-dataset --help
```

### Available Commands

#### 1. Download Datasets

Download public datasets for license plate character recognition:

```bash
# Download EMNIST dataset (default)
ipsem2025-dataset download

# Specify output directory
ipsem2025-dataset download --output-dir ./my_datasets

# Force re-download
ipsem2025-dataset download --force
```

Options:
- `--dataset-name, -d`: Dataset to download (currently supports "emnist")
- `--output-dir, -o`: Directory to store the dataset (default: "data")
- `--force, -f`: Force re-download even if dataset exists
- `--verbose, -v`: Enable verbose output

#### 2. Display Dataset Information

View detailed information about a dataset:

```bash
# Basic usage
ipsem2025-dataset info --dataset-path ./data

# Specify dataset type explicitly
ipsem2025-dataset info --dataset-path ./my_custom_data --dataset-type custom
```

Options:
- `--dataset-path, -p`: Path to the dataset (required)
- `--dataset-type, -t`: Type of dataset (auto, emnist, custom)
- `--verbose, -v`: Enable verbose output

This command shows:
- Number of samples and classes
- Class mapping and distribution
- Image dimensions

#### 3. Preview Dataset Samples

Visualize sample images from a dataset:

```bash
# Preview with default settings
ipsem2025-dataset preview --dataset-path ./data

# Show more samples
ipsem2025-dataset preview --dataset-path ./data --num-samples 12

# Save preview to file instead of displaying
ipsem2025-dataset preview --dataset-path ./data --save-plot preview.png
```

Options:
- `--dataset-path, -p`: Path to the dataset (required)
- `--dataset-type, -t`: Type of dataset (auto, emnist, custom)
- `--num-samples, -n`: Number of samples to display (default: 8)
- `--save-plot, -s`: Save preview to file instead of displaying
- `--verbose, -v`: Enable verbose output

#### 4. Validate Custom Datasets

Check if a custom dataset follows the expected structure:

```bash
# Validate a custom dataset
ipsem2025-dataset validate --dataset-path ./my_custom_data
```

Options:
- `--dataset-path, -p`: Path to the dataset to validate (required)
- `--fix, -f`: Attempt to fix common issues
- `--verbose, -v`: Enable verbose output

This command checks:
- Directory structure (one folder per class)
- Image formats and validity
- Missing or corrupted files

### Common Options

All commands support the following logging options:

- `--log-file`: Path to log file. If not specified, logs to console only
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--verbose, -v`: Enable verbose output (same as --log-level DEBUG)

Example with logging options:
```bash
# Log to both console and file with DEBUG level
ipsem2025-dataset download --log-file ./logs/dataset.log --log-level DEBUG

# Use verbose mode (equivalent to DEBUG level)
ipsem2025-dataset preview --dataset-path ./data -v

# Specify custom log level
ipsem2025-dataset info --dataset-path ./data --log-level WARNING
```

## Usage Examples (Python API)

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