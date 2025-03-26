"""Tests for the BaseDataset abstract class implementation.

Tests core functionality and interface requirements that all dataset implementations
must satisfy. Ensures compatibility with the training pipeline through:

1. Base Properties: Size, dimensions, class count
2. Data Access: Image tensors, label formats
3. Class Mappings: Index to label conversion
4. Data Loading: Train/val/test splits
5. Factory Methods: Path-based creation

Passing tests indicate the dataset interface is well-defined and maintains
type safety, proper error handling, and PyTorch compatibility.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from ipsem2025_license_plate.datasets.base import BaseDataset
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DummyDataset(BaseDataset):
    """A minimal concrete implementation of BaseDataset for testing."""

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self._images = torch.randn(num_samples, 1, 28, 28)  # Simple random images
        self._labels = torch.randint(0, 10, (num_samples,))  # Random labels 0-9

    def get_image_dimensions(self):
        return (1, 28, 28)  # Single channel 28x28 images

    def get_num_classes(self):
        return 10

    def get_class_mapping(self):
        return {i: str(i) for i in range(10)}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._images[idx], self._labels[idx].item()

    @classmethod
    def from_path(cls, path, **kwargs):
        return cls(**kwargs)


def test_dataset_creation():
    """Test basic dataset creation."""
    logger.info("Testing basic dataset creation...")
    dataset = DummyDataset(num_samples=100)
    logger.info("Created dummy dataset with %d samples", len(dataset))
    assert len(dataset) == 100
    assert dataset.get_num_classes() == 10
    assert dataset.get_image_dimensions() == (1, 28, 28)
    logger.info("Dataset creation test passed successfully")


def test_class_mapping():
    """Test class mapping functionality."""
    logger.info("Testing class mapping functionality...")
    dataset = DummyDataset()
    mapping = dataset.get_class_mapping()
    logger.info("Retrieved class mapping with %d entries", len(mapping))
    assert len(mapping) == 10
    assert all(isinstance(k, int) for k in mapping.keys())
    assert all(isinstance(v, str) for v in mapping.values())
    logger.info("Class mapping test passed successfully")


def test_data_access():
    """Test accessing individual items."""
    logger.info("Testing data access functionality...")
    dataset = DummyDataset()
    logger.info("Accessing first item in dataset")
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 28, 28)
    assert isinstance(label, int)
    assert 0 <= label < 10
    logger.info(
        "Data access test passed successfully. Image shape: %s, Label: %d",
        str(image.shape),
        label,
    )


def test_data_loaders():
    """Test creation of data loaders."""
    logger.info("Testing data loader creation...")
    dataset = DummyDataset(num_samples=100)
    logger.info("Creating data loaders with 70/20/10 split")
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.2,
        num_workers=0,  # Use 0 workers for testing
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check split sizes
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    assert train_size == 70  # 70% of 100
    assert val_size == 20  # 20% of 100
    assert test_size == 10  # Remaining 10%

    logger.info(
        "Data loader creation successful - Train: %d, Val: %d, Test: %d samples",
        train_size,
        val_size,
        test_size,
    )


def test_from_path():
    """Test dataset creation from path."""
    logger.info("Testing dataset creation from path...")
    dataset = DummyDataset.from_path("dummy/path", num_samples=50)
    assert len(dataset) == 50
    logger.info("Dataset creation from path successful with %d samples", len(dataset))


def test_abstract_class():
    """Test that BaseDataset cannot be instantiated directly."""
    logger.info("Testing that BaseDataset cannot be instantiated directly...")
    with pytest.raises(TypeError):
        BaseDataset()
    logger.info("Abstract class test passed successfully")
