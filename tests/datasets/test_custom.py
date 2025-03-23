"""Tests for the CustomImageDataset implementation.

Tests loading and processing of custom image datasets organized in class folders.
Validates:

1. Directory Structure: Class folders, image formats, auto-discovery
2. Image Processing: Loading, resizing, tensor conversion
3. Error Handling: Invalid paths, empty dirs, bad formats
4. Class Mapping: Directory names to indices
5. Data Loading: Batch processing, transforms

Passing tests confirm reliable handling of user-provided image collections
with proper error handling and memory efficiency.
"""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from ipsem2025_license_plate.datasets.custom import CustomImageDataset
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def test_images_root(tmp_path_factory):
    """Create a temporary directory with test images."""
    root = tmp_path_factory.mktemp("custom_dataset")
    logger.info("Creating test dataset structure at: %s", root)

    # Create class directories
    classes = ["0", "1", "A", "B"]  # Mix of numbers and letters
    colors = ["#ff0000", "#00ff00", "#0000ff"]  # Red, Green, Blue

    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir()
        logger.info("Created class directory: %s", class_dir)

        # Create test images for each class
        for i, color in enumerate(colors):
            img = Image.new("RGB", (32, 32), color=color)
            img_path = class_dir / f"img_{i}.png"
            img.save(img_path)
            logger.info("Created test image: %s", img_path)

    return root


@pytest.fixture(scope="session")
def base_dataset(test_images_root):
    """Create the base custom dataset instance used by tests."""
    logger.info("Initializing base CustomImageDataset...")
    dataset = CustomImageDataset(root=str(test_images_root))
    logger.info("Dataset initialized with %d samples", len(dataset))
    return dataset


@pytest.fixture(scope="session")
def class_mapping(base_dataset):
    """Get the class mapping from the dataset."""
    mapping = base_dataset.get_class_mapping()
    logger.info("Retrieved class mapping with %d entries", len(mapping))
    return mapping


@pytest.fixture(scope="session")
def data_loaders(base_dataset):
    """Create data loaders once for all tests."""
    logger.info("Creating data loaders...")
    loaders = base_dataset.create_data_loaders(
        batch_size=2,
        train_ratio=0.5,
        val_ratio=0.25,
        num_workers=0,  # Use 0 for testing
    )
    logger.info(
        "Data loaders created with sizes: train=%d, val=%d, test=%d",
        len(loaders[0].dataset),
        len(loaders[1].dataset),
        len(loaders[2].dataset),
    )
    return loaders


def test_dataset_creation(base_dataset):
    """Test basic dataset creation and properties."""
    logger.info("Testing dataset creation...")
    assert isinstance(base_dataset, CustomImageDataset)
    assert len(base_dataset) == 12  # 4 classes * 3 images
    logger.info("Dataset creation test passed with %d samples", len(base_dataset))


def test_image_dimensions(base_dataset):
    """Test image dimensions match expected format."""
    logger.info("Testing image dimensions...")
    dims = base_dataset.get_image_dimensions()
    assert dims == (3, 64, 64)  # RGB images, resized to 64x64

    img, _ = base_dataset[0]
    assert img.shape == dims
    logger.info(
        "Image dimensions test passed. Images are %dx%d with %d channels",
        dims[1],
        dims[2],
        dims[0],
    )


def test_num_classes(base_dataset):
    """Test number of classes is correct."""
    logger.info("Testing number of classes...")
    num_classes = base_dataset.get_num_classes()
    assert num_classes == 4  # ['0', '1', 'A', 'B']
    logger.info("Number of classes test passed with %d classes", num_classes)


def test_class_mapping(class_mapping):
    """Test class mapping functionality."""
    logger.info("Testing class mapping...")
    assert len(class_mapping) == 4
    assert all(isinstance(k, int) for k in class_mapping.keys())
    assert all(isinstance(v, str) for v in class_mapping.values())
    assert set(class_mapping.values()) == {"0", "1", "A", "B"}
    logger.info(
        "Class mapping test passed with classes: %s", ", ".join(class_mapping.values())
    )


def test_data_access(base_dataset):
    """Test accessing individual items."""
    logger.info("Testing data access...")
    # Test first item
    img, label = base_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 4

    # Test random access
    idx = len(base_dataset) // 2
    img, label = base_dataset[idx]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 4
    logger.info("Data access test passed for first and middle items")


def test_data_loaders(data_loaders):
    """Test creation of data loaders."""
    logger.info("Testing data loader properties...")
    train_loader, val_loader, test_loader = data_loaders

    # Verify loader types
    assert all(
        isinstance(loader, torch.utils.data.DataLoader)
        for loader in [train_loader, val_loader, test_loader]
    )

    # Check split sizes (12 total images)
    assert len(train_loader.dataset) == 6  # 50%
    assert len(val_loader.dataset) == 3  # 25%
    assert len(test_loader.dataset) == 3  # 25%

    logger.info(
        "Data loader test passed. Split sizes - Train: %d, Val: %d, Test: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )


def test_custom_transform(test_images_root):
    """Test dataset with custom transform."""
    logger.info("Testing custom transform...")
    custom_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Different size
            transforms.ToTensor(),
        ]
    )

    logger.info("Creating dataset with custom 32x32 transform...")
    dataset = CustomImageDataset(root=str(test_images_root), transform=custom_transform)

    img, _ = dataset[0]
    assert img.shape == (3, 32, 32)
    logger.info(
        "Custom transform test passed with image size %dx%d", img.shape[1], img.shape[2]
    )


def test_from_path(test_images_root):
    """Test dataset creation from path."""
    logger.info("Testing dataset creation from path: %s", test_images_root)
    dataset = CustomImageDataset.from_path(str(test_images_root))
    assert isinstance(dataset, CustomImageDataset)
    assert len(dataset) == 12
    logger.info("Dataset creation from path test passed with %d samples", len(dataset))


def test_invalid_directory():
    """Test dataset creation with invalid directory."""
    logger.info("Testing invalid directory handling...")
    with pytest.raises(ValueError) as exc_info:
        CustomImageDataset(root="nonexistent_directory")
    assert "Dataset root directory does not exist" in str(exc_info.value)
    logger.info("Invalid directory test passed with expected error")


def test_empty_class_directory(tmp_path):
    """Test dataset creation with empty class directories."""
    logger.info("Testing empty class directory handling...")
    # Create empty class directories
    (tmp_path / "class1").mkdir()
    (tmp_path / "class2").mkdir()

    with pytest.raises(ValueError) as exc_info:
        CustomImageDataset(root=str(tmp_path))
    assert "No valid images found" in str(exc_info.value)
    logger.info("Empty class directory test passed with expected error")
