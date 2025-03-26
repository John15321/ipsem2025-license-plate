"""
Global pytest fixtures available to all tests.

This module provides fixtures that are automatically available to all test modules.
It focuses on resource-intensive fixtures like datasets that should be reused across tests.
"""

# pylint: disable=redefined-outer-name

import pytest
from torchvision import transforms

from ipsem2025_license_plate.datasets.emnist import EMNISTDataset
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)

# pylint: disable=redefined-outer-name
@pytest.fixture(scope="session")
def global_temp_dir(tmp_path_factory):
    """Create a temporary directory that persists across all tests."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    logger.info("Created global temporary directory: %s", temp_dir)
    yield temp_dir
    # Optional cleanup after all tests complete
    # shutil.rmtree(temp_dir)

# pylint: disable=redefined-outer-name
@pytest.fixture(scope="session")
def emnist_dataset_root(global_temp_dir):
    """Create a temporary directory for EMNIST dataset storage that persists across all tests."""
    root = global_temp_dir / "emnist_data"
    root.mkdir(exist_ok=True)
    logger.info("Created EMNIST dataset root at: %s", root)
    return root

# pylint: disable=redefined-outer-name
@pytest.fixture(scope="session")
def emnist_downloaded_dataset(emnist_dataset_root):
    """
    Download EMNIST dataset once and share it across all tests.

    This is a resource-intensive operation that should only happen once
    during the entire test session.
    """
    logger.info("Downloading EMNIST dataset (this will only happen once)...")
    dataset = EMNISTDataset(root=str(emnist_dataset_root), download=True)
    logger.info("EMNIST dataset downloaded successfully with %d samples", len(dataset))
    return dataset

# pylint: disable=redefined-outer-name
@pytest.fixture(scope="session")
def emnist_transform():
    """Default transform for EMNIST dataset tests."""
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),  # invert colors
        ]
    )


@pytest.fixture(scope="session")
def emnist_class_mapping(emnist_downloaded_dataset):
    """Get the class mapping from the dataset once for all tests."""
    mapping = emnist_downloaded_dataset.get_class_mapping()
    logger.info("Retrieved EMNIST class mapping with %d entries", len(mapping))
    return mapping
