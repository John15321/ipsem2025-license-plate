"""Base interface for all datasets used in the project."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets in the project.

    This class defines the interface that all dataset implementations
    must follow to ensure compatibility with the training pipeline.
    """

    @abstractmethod
    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of images in the dataset.

        Returns:
            Tuple of (channels, height, width)
        """
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset.

        Returns:
            Number of unique classes
        """
        pass

    @abstractmethod
    def get_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from class indices to class names.

        Returns:
            Dictionary mapping class indices to human-readable names
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image, label) where image is a tensor and
            label is an integer class index
        """
        pass

    @classmethod
    @abstractmethod
    def from_path(cls, path: str, **kwargs: Any) -> "BaseDataset":
        """Create a dataset instance from a filesystem path.

        This method should handle loading and validating the dataset
        from various supported formats (folders, archives, etc).

        Args:
            path: Path to the dataset files
            **kwargs: Additional dataset-specific parameters

        Returns:
            Initialized dataset instance

        Raises:
            ValueError: If the dataset at the path is invalid or unsupported
        """
        pass

    def create_data_loaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test data loaders.

        Args:
            batch_size: Number of samples per batch
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            num_workers: Number of worker processes for data loading
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        total_size = len(self)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader
