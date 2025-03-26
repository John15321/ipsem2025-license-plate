"""Tests for the MNIST dataset implementation.

Tests the MNIST dataset implementation for digit recognition.
Validates:

1. Data Setup: Auto-download, extraction, caching
2. Image Processing: 64x64 resizing
3. Label Handling: 10 classes (0-9)
4. Data Loading: Custom transforms, deterministic batching
5. Dataset Interface: Compliance with BaseDataset
"""

import os
from unittest import mock

import pytest
import torch
from torchvision import transforms

from ipsem2025_license_plate.datasets.mnist import MNISTDataset
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def mnist_dataset_root(tmp_path_factory):
    """Create a temporary directory for dataset storage."""
    return tmp_path_factory.mktemp("mnist_data")


@pytest.fixture(scope="session")
def mnist_downloaded_dataset(mnist_dataset_root):
    """Create and cache a dataset instance for all tests."""
    logger.info("Setting up MNIST dataset at %s", mnist_dataset_root)
    dataset = MNISTDataset(root=str(mnist_dataset_root), download=True)
    logger.info("MNIST dataset initialized with %d samples", len(dataset))
    return dataset


@pytest.fixture(scope="session")
def mnist_class_mapping(mnist_downloaded_dataset):
    """Get the class mapping once for all tests that need it."""
    return mnist_downloaded_dataset.get_class_mapping()


@pytest.fixture(scope="session")
def data_loaders(mnist_downloaded_dataset):
    """Create data loaders once for all tests that need them."""
    logger.info("Creating data loaders from global dataset instance...")
    loaders = mnist_downloaded_dataset.create_data_loaders(
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.2,
        num_workers=0,  # Use 0 for testing
    )
    logger.info(
        "Data loaders created with sizes: train=%d, val=%d, test=%d",
        len(loaders[0].dataset),
        len(loaders[1].dataset),
        len(loaders[2].dataset),
    )
    return loaders


def test_dataset_creation(mnist_downloaded_dataset):
    """Test basic dataset creation and properties."""
    logger.info("Testing dataset creation...")
    assert isinstance(mnist_downloaded_dataset, MNISTDataset)
    assert len(mnist_downloaded_dataset) > 0
    logger.info(
        "Dataset creation test passed with %d samples", len(mnist_downloaded_dataset)
    )


def test_image_dimensions(mnist_downloaded_dataset):
    """Test image dimensions match expected format."""
    logger.info("Testing image dimensions...")
    dims = mnist_downloaded_dataset.get_image_dimensions()
    assert dims == (1, 64, 64)  # Single channel, 64x64 images

    img, _ = mnist_downloaded_dataset[0]
    assert img.shape == dims
    logger.info(
        "Image dimensions test passed. Images are %dx%d with %d channels",
        dims[1],
        dims[2],
        dims[0],
    )


def test_num_classes(mnist_downloaded_dataset):
    """Test number of classes is correct."""
    logger.info("Testing number of classes...")
    num_classes = mnist_downloaded_dataset.get_num_classes()
    assert num_classes == 10  # digits 0-9
    logger.info("Number of classes test passed with %d classes", num_classes)


def test_class_mapping(mnist_class_mapping):
    """Test class mapping functionality."""
    logger.info("Testing class mapping...")
    # Check mapping properties
    assert len(mnist_class_mapping) == 10
    assert all(isinstance(k, int) for k in mnist_class_mapping.keys())
    assert all(isinstance(v, str) for v in mnist_class_mapping.values())

    # Check specific mappings
    for i in range(10):
        assert mnist_class_mapping[i] == str(i), f"Incorrect mapping for digit {i}"

    logger.info("Class mapping test passed with all 10 digits verified")


def test_data_access(mnist_downloaded_dataset):
    """Test accessing individual items."""
    logger.info("Testing data access...")
    # Test first item
    img, label = mnist_downloaded_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 10

    # Test random access
    idx = len(mnist_downloaded_dataset) // 2
    img, label = mnist_downloaded_dataset[idx]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 10
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

    # Check split sizes sum to total
    total_size = sum(
        len(loader.dataset) for loader in [train_loader, val_loader, test_loader]
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    assert train_size == int(0.7 * total_size)
    assert val_size == int(0.2 * total_size)
    assert test_size == total_size - train_size - val_size

    logger.info(
        "Data loader test passed. Split sizes - Train: %d, Val: %d, Test: %d",
        train_size,
        val_size,
        test_size,
    )


def test_custom_transform(mnist_dataset_root, mnist_downloaded_dataset):
    """Test dataset with custom transform."""
    logger.info("Testing custom transform...")
    custom_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Different size
            transforms.ToTensor(),
        ]
    )

    logger.info(
        "Creating dataset with custom 32x32 transform (reusing existing data)..."
    )
    dataset = MNISTDataset(
        root=str(mnist_dataset_root),
        transform=custom_transform,
        download=False,  # Important: don't download again
    )

    img, _ = dataset[0]
    assert img.shape == (1, 32, 32)
    logger.info(
        "Custom transform test passed with image size %dx%d", img.shape[1], img.shape[2]
    )


def test_from_path(mnist_dataset_root, mnist_downloaded_dataset):
    """Test dataset creation from path."""
    logger.info("Testing dataset creation from path: %s", mnist_dataset_root)
    dataset = MNISTDataset.from_path(
        str(mnist_dataset_root), download=False  # Reuse existing data
    )
    assert isinstance(dataset, MNISTDataset)
    assert len(dataset) > 0
    logger.info("Dataset creation from path test passed with %d samples", len(dataset))


def test_exists_at_path(mnist_dataset_root):
    """Test exists_at_path static method."""
    logger.info("Testing exists_at_path functionality...")

    # First ensure dataset exists by creating an instance
    _ = MNISTDataset(root=str(mnist_dataset_root), download=True)

    # Test with existing dataset
    assert MNISTDataset.exists_at_path(str(mnist_dataset_root))

    # Test with non-existent path
    assert not MNISTDataset.exists_at_path("/nonexistent/path")

    logger.info("exists_at_path test passed")


def test_mock_dataset():
    """Test using a mock to avoid downloads completely."""
    with mock.patch("torchvision.datasets.MNIST") as mock_mnist:
        # Configure the mock to return predictable data
        mock_instance = mock.MagicMock()
        mock_instance.__len__.return_value = 100
        mock_instance.__getitem__.side_effect = lambda idx: (
            torch.randn(1, 64, 64),  # Random tensor as image
            idx % 10,  # Cycle through 0-9 as labels
        )
        mock_mnist.return_value = mock_instance

        # Create dataset with mocked components
        dataset = MNISTDataset(root="mock_path", download=False)
        dataset.dataset = mock_instance  # Set mocked dataset

        # Test that the dataset works correctly with mocked components
        assert len(dataset) == 100
        img, label = dataset[0]
        assert img.shape == (1, 64, 64)
        assert 0 <= label < 10

    logger.info("Mock dataset test passed - successfully tested without downloads")
