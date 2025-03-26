"""MNIST dataset implementation."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
from torchvision import datasets, transforms

from ..utils.logging_utils import get_logger
from .base import BaseDataset

logger = get_logger(__name__)


class MNISTDataset(BaseDataset):
    """MNIST dataset filtered to digits only."""

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ):
        super().__init__()
        self.root = root
        self.train = train

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

        # Load MNIST dataset
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

        logger.info(
            "Initialized MNISTDataset with %d samples (digits 0-9)",
            len(self.dataset),
        )

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of images in the dataset."""
        # MNIST images are grayscale 64x64 after our transform
        return (1, 64, 64)

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return 10  # digits 0-9

    def get_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from class indices to class names."""
        return {i: str(i) for i in range(10)}

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        return self.dataset[idx]

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "MNISTDataset":
        """Create a dataset instance from a filesystem path."""
        return cls(root=path, **kwargs)

    @staticmethod
    def exists_at_path(path: str) -> bool:
        """Check if MNIST dataset exists at the given path.

        Args:
            path: Path to check for dataset

        Returns:
            True if dataset exists, False otherwise
        """
        # MNIST files that should exist after download
        required_files = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ]

        # Check for processed files
        raw_folder = os.path.join(path, "MNIST", "raw")
        return all(
            os.path.exists(os.path.join(raw_folder, f))
            or os.path.exists(os.path.join(raw_folder, f + ".gz"))
            for f in required_files
        )
