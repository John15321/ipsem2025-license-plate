"""MNIST dataset implementation."""

from typing import Any, Dict, Optional, Tuple

import torch
from torchvision import datasets, transforms

from ..utils.logging_utils import get_logger
from .base import BaseDataset

logger = get_logger(__name__)


class MNISTDataset(BaseDataset):
    """MNIST dataset wrapper implementing BaseDataset interface."""

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ):
        """Initialize the MNIST dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to load training or test set
            transform: Optional transform to apply to images
            download: Whether to download the dataset if not found
        """
        super().__init__()
        self.root = root
        self.train = train
        self.download = download

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        self.transform = transform

        logger.info("Loading MNIST dataset")
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        logger.info(f"Loaded MNIST dataset with {len(self.dataset)} samples")

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of images in the dataset."""
        return (1, 64, 64)  # Grayscale 64x64 after resize

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return 10  # MNIST has 10 classes (0-9)

    def get_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from class indices to class names."""
        return {i: str(i) for i in range(10)}

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        img, label = self.dataset[idx]
        return img, label

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "MNISTDataset":
        """Create a dataset instance from a filesystem path."""
        return cls(root=path, **kwargs)

    @staticmethod
    def exists_at_path(path: str) -> bool:
        """Check if a MNIST dataset exists at the given path."""
        import os
        processed_folder = os.path.join(path, "MNIST", "processed")
        return os.path.exists(processed_folder) and len(os.listdir(processed_folder)) > 0