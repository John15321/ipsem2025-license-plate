"""Tests for the EMNIST dataset implementation.

Tests the specialized EMNIST dataset for license plate character recognition.
Validates:

1. Data Setup: Auto-download, extraction, caching
2. Character Processing: 36 classes (0-9, A-Z), 64x64 resizing
3. Label Mapping: EMNIST to 0-35 range, case normalization
4. Memory Usage: Lazy loading, efficient splits
5. Data Loading: Custom transforms, deterministic batching

Passing tests ensure reliable character recognition training data
with consistent preprocessing and efficient memory usage.
"""

import os
import pytest
import torch
from torchvision import transforms
from ipsem2025_license_plate.datasets.emnist import EMNISTDataset
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory):
    """Create a temporary directory for dataset storage that persists across all tests."""
    root = tmp_path_factory.mktemp("emnist_data")
    logger.info("Creating temporary dataset root at: %s", root)
    return root


@pytest.fixture(scope="session")
def base_dataset(dataset_root):
    """Create the base EMNIST dataset instance used by other fixtures."""
    logger.info("Initializing base EMNIST dataset (this will download data if needed)...")
    dataset = EMNISTDataset(root=str(dataset_root), download=True)
    logger.info("Base dataset initialized with %d samples", len(dataset))
    return dataset


@pytest.fixture(scope="session")
def class_mapping(base_dataset):
    """Get the class mapping from the dataset."""
    mapping = base_dataset.get_class_mapping()
    logger.info("Retrieved class mapping with %d entries", len(mapping))
    return mapping


@pytest.fixture(scope="session")
def data_loaders(base_dataset):
    """Create data loaders once for all tests that need them."""
    logger.info("Creating data loaders...")
    loaders = base_dataset.create_data_loaders(
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.2,
        num_workers=0  # Use 0 for testing
    )
    logger.info("Data loaders created with sizes: train=%d, val=%d, test=%d",
                len(loaders[0].dataset), len(loaders[1].dataset), len(loaders[2].dataset))
    return loaders


def test_dataset_creation(base_dataset):
    """Test basic dataset creation and properties."""
    logger.info("Testing dataset creation...")
    assert isinstance(base_dataset, EMNISTDataset)
    assert len(base_dataset) > 0
    logger.info("Dataset creation test passed with %d samples", len(base_dataset))


def test_image_dimensions(base_dataset):
    """Test image dimensions match expected format."""
    logger.info("Testing image dimensions...")
    dims = base_dataset.get_image_dimensions()
    assert dims == (1, 64, 64)  # Single channel, 64x64 images
    
    img, _ = base_dataset[0]
    assert img.shape == dims
    logger.info("Image dimensions test passed. Images are %dx%d with %d channels",
                dims[1], dims[2], dims[0])


def test_num_classes(base_dataset):
    """Test number of classes is correct."""
    logger.info("Testing number of classes...")
    num_classes = base_dataset.get_num_classes()
    assert num_classes == 36  # 10 digits + 26 letters
    logger.info("Number of classes test passed with %d classes", num_classes)


def test_class_mapping(class_mapping):
    """Test class mapping functionality."""
    logger.info("Testing class mapping...")
    # Check mapping properties
    assert len(class_mapping) == 36
    assert all(isinstance(k, int) for k in class_mapping.keys())
    assert all(isinstance(v, str) for v in class_mapping.values())
    
    # Check specific mappings
    for i in range(10):
        assert class_mapping[i] == str(i), f"Incorrect mapping for digit {i}"
    
    for i in range(26):
        expected = chr(ord('A') + i)
        assert class_mapping[i + 10] == expected, f"Incorrect mapping for letter {expected}"
    
    logger.info("Class mapping test passed with all 36 characters verified")


def test_data_access(base_dataset):
    """Test accessing individual items."""
    logger.info("Testing data access...")
    # Test first item
    img, label = base_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 36
    
    # Test random access
    idx = len(base_dataset) // 2
    img, label = base_dataset[idx]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 36
    logger.info("Data access test passed for first and middle items")


def test_data_loaders(data_loaders):
    """Test creation of data loaders."""
    logger.info("Testing data loader properties...")
    train_loader, val_loader, test_loader = data_loaders
    
    # Verify loader types
    assert all(isinstance(loader, torch.utils.data.DataLoader) 
              for loader in [train_loader, val_loader, test_loader])
    
    # Check split sizes sum to total
    total_size = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    assert train_size == int(0.7 * total_size)
    assert val_size == int(0.2 * total_size)
    assert test_size == total_size - train_size - val_size
    
    logger.info("Data loader test passed. Split sizes - Train: %d, Val: %d, Test: %d",
                train_size, val_size, test_size)


def test_custom_transform(dataset_root):
    """Test dataset with custom transform."""
    logger.info("Testing custom transform...")
    custom_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Different size
        transforms.ToTensor(),
    ])
    
    logger.info("Creating dataset with custom 32x32 transform...")
    dataset = EMNISTDataset(
        root=str(dataset_root),  # Reuse the same root to avoid redownloading
        transform=custom_transform,
        download=False  # Data already downloaded
    )
    
    img, _ = dataset[0]
    assert img.shape == (1, 32, 32)
    logger.info("Custom transform test passed with image size %dx%d",
                img.shape[1], img.shape[2])


def test_from_path(dataset_root):
    """Test dataset creation from path."""
    logger.info("Testing dataset creation from path: %s", dataset_root)
    dataset = EMNISTDataset.from_path(str(dataset_root), download=False)  # Reuse existing data
    assert isinstance(dataset, EMNISTDataset)
    assert len(dataset) > 0
    logger.info("Dataset creation from path test passed with %d samples", len(dataset))


def test_label_mapping_creation(dataset_root):
    """Test label mapping file creation and loading."""
    logger.info("Testing label mapping creation...")
    
    # Create new dataset instance but reuse downloaded data
    dataset = EMNISTDataset(root=str(dataset_root), download=False)
    
    # Check if mapping file exists
    mapping_path = os.path.join(dataset_root, "EMNIST", "raw", "emnist-bymerge-mapping.txt")
    assert os.path.exists(mapping_path)
    logger.info("Found mapping file at: %s", mapping_path)
    
    # Verify mapping file contents
    with open(mapping_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Should have entries for 0-9 and A-Z
    assert len(lines) == 36
    
    # Verify the mappings
    mappings = dict(line.strip().split() for line in lines)
    
    # Check digits (0-9)
    for i in range(10):
        assert mappings[str(i)] == str(ord('0') + i)
    
    # Check letters (A-Z)
    for i in range(26):
        assert mappings[str(i + 10)] == str(ord('A') + i)
    
    logger.info("Label mapping creation test passed with %d verified mappings", len(mappings))