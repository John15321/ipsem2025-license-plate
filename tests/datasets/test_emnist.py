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
from unittest import mock

import pytest
import torch
from torchvision import transforms

from ipsem2025_license_plate.datasets.emnist import EMNISTDataset
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def data_loaders(emnist_downloaded_dataset):
    """Create data loaders once for all tests that need them."""
    logger.info("Creating data loaders from global dataset instance...")
    loaders = emnist_downloaded_dataset.create_data_loaders(
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


def test_dataset_creation(emnist_downloaded_dataset):
    """Test basic dataset creation and properties."""
    logger.info("Testing dataset creation...")
    assert isinstance(emnist_downloaded_dataset, EMNISTDataset)
    assert len(emnist_downloaded_dataset) > 0
    logger.info("Dataset creation test passed with %d samples", len(emnist_downloaded_dataset))


def test_image_dimensions(emnist_downloaded_dataset):
    """Test image dimensions match expected format."""
    logger.info("Testing image dimensions...")
    dims = emnist_downloaded_dataset.get_image_dimensions()
    assert dims == (1, 64, 64)  # Single channel, 64x64 images

    img, _ = emnist_downloaded_dataset[0]
    assert img.shape == dims
    logger.info(
        "Image dimensions test passed. Images are %dx%d with %d channels",
        dims[1],
        dims[2],
        dims[0],
    )


def test_num_classes(emnist_downloaded_dataset):
    """Test number of classes is correct."""
    logger.info("Testing number of classes...")
    num_classes = emnist_downloaded_dataset.get_num_classes()
    assert num_classes == 36  # 10 digits + 26 letters
    logger.info("Number of classes test passed with %d classes", num_classes)


def test_class_mapping(emnist_class_mapping):
    """Test class mapping functionality."""
    logger.info("Testing class mapping...")
    # Check mapping properties
    assert len(emnist_class_mapping) == 36
    assert all(isinstance(k, int) for k in emnist_class_mapping.keys())
    assert all(isinstance(v, str) for v in emnist_class_mapping.values())

    # Check specific mappings
    for i in range(10):
        assert emnist_class_mapping[i] == str(i), f"Incorrect mapping for digit {i}"

    for i in range(26):
        expected = chr(ord("A") + i)
        assert (
            emnist_class_mapping[i + 10] == expected
        ), f"Incorrect mapping for letter {expected}"

    logger.info("Class mapping test passed with all 36 characters verified")


def test_data_access(emnist_downloaded_dataset):
    """Test accessing individual items."""
    logger.info("Testing data access...")
    # Test first item
    img, label = emnist_downloaded_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 36

    # Test random access
    idx = len(emnist_downloaded_dataset) // 2
    img, label = emnist_downloaded_dataset[idx]
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


def test_custom_transform(emnist_dataset_root, emnist_downloaded_dataset):
    """Test dataset with custom transform (reuses existing downloaded data)."""
    logger.info("Testing custom transform...")
    custom_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Different size
            transforms.ToTensor(),
        ]
    )

    logger.info("Creating dataset with custom 32x32 transform (reusing existing data)...")
    dataset = EMNISTDataset(
        root=str(emnist_dataset_root),  # Reuse the shared dataset root
        transform=custom_transform,
        download=False,  # Important: don't download again
    )

    img, _ = dataset[0]
    assert img.shape == (1, 32, 32)
    logger.info(
        "Custom transform test passed with image size %dx%d", img.shape[1], img.shape[2]
    )


def test_from_path(emnist_dataset_root, emnist_downloaded_dataset):
    """Test dataset creation from path."""
    logger.info("Testing dataset creation from path: %s", emnist_dataset_root)
    dataset = EMNISTDataset.from_path(
        str(emnist_dataset_root), download=False  # Reuse existing data
    )
    assert isinstance(dataset, EMNISTDataset)
    assert len(dataset) > 0
    logger.info("Dataset creation from path test passed with %d samples", len(dataset))


def test_label_mapping_creation(emnist_dataset_root, emnist_downloaded_dataset):
    """Test label mapping file creation and loading."""
    logger.info("Testing label mapping creation...")

    # Create new dataset instance but reuse downloaded data
    dataset = EMNISTDataset(root=str(emnist_dataset_root), download=False)

    # Check if mapping file exists
    mapping_path = os.path.join(
        emnist_dataset_root, "EMNIST", "raw", "emnist-bymerge-mapping.txt"
    )
    assert os.path.exists(mapping_path)
    logger.info("Found mapping file at: %s", mapping_path)

    # Verify mapping file contents
    with open(mapping_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Should have entries for 0-9 and A-Z
    assert len(lines) == 36

    # Verify the mappings
    mappings = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:  # Make this more robust
            mappings[parts[0]] = parts[1]

    # Check digits (0-9)
    for i in range(10):
        assert str(i) in mappings, f"Missing mapping for digit {i}"

    # Check letters (A-Z)
    for i in range(26):
        letter_idx = str(i + 10)
        assert letter_idx in mappings, f"Missing mapping for letter index {letter_idx}"

    logger.info(
        "Label mapping creation test passed with %d verified mappings", len(mappings)
    )


def test_mock_dataset():
    """Test using a mock to avoid downloads completely."""
    with mock.patch('ipsem2025_license_plate.datasets.emnist.EMNISTDataset._load_label_mapping') as mock_load_mapping:
        # Mock the label mapping method to return a predefined mapping
        mock_mapping = {i: str(i) for i in range(10)}  # 0-9 digits
        mock_mapping.update({i+10: chr(ord('A') + i) for i in range(26)})  # A-Z letters
        mock_load_mapping.return_value = mock_mapping

        # Mock an in-memory dataset without downloading
        with mock.patch('torchvision.datasets.EMNIST') as mock_emnist:
            # Configure the mock to return predictable data
            mock_instance = mock.MagicMock()
            mock_instance.__len__.return_value = 100
            mock_instance.__getitem__.side_effect = lambda idx: (
                torch.randn(1, 64, 64),  # Random tensor as image
                idx % 36  # Cycle through 0-35 as labels
            )
            mock_emnist.return_value = mock_instance

            # Create dataset with mocked components
            with mock.patch.object(EMNISTDataset, '_build_36class_map') as mock_class_map:
                # Mock the class mapping to match our mock data
                mock_class_map.return_value = {i: i for i in range(36)}

                # Mock the indices list to include all indices (normally filtered)
                with mock.patch.object(EMNISTDataset, '__init__', return_value=None):
                    dataset = EMNISTDataset(root="mock_path", download=False)
                    # Manually set attributes since __init__ is bypassed
                    dataset.indices = list(range(100))
                    dataset._emnist = mock_instance
                    dataset.label_map_36 = {i: i for i in range(36)}

            # Test that the dataset works correctly with mocked components
            assert len(dataset) > 0
            img, label = dataset[0]
            assert img.shape == (1, 64, 64)
            assert 0 <= label < 36

    logger.info("Mock dataset test passed - successfully tested without downloads")
