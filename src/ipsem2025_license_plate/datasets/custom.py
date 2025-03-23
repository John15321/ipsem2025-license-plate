"""Custom dataset implementation for folder-based image datasets."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from ..utils.logging_utils import get_logger
from .base import BaseDataset

logger = get_logger(__name__)


class CustomImageDataset(BaseDataset):
    """Dataset for loading images organized in class folders.

    Expected directory structure:
    dataset_root/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            ...
        ...
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize the custom dataset.

        Args:
            root: Root directory containing class folders
            transform: Optional transform to apply to images
        """
        super().__init__()
        self.root = Path(root)

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

        # Find all image files and build class mapping
        self.files: List[Path] = []
        self.class_to_idx: Dict[str, int] = {}
        self._scan_directory()

        if not self.files:
            raise ValueError(
                f"No valid images found in {root}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        logger.info(
            "Initialized CustomImageDataset with %d images in %d classes from %s",
            len(self.files),
            len(self.class_to_idx),
            root,
        )

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of images in the dataset."""
        # Open first image to get dimensions
        with Image.open(self.files[0]) as img:
            if self.transform:
                img = self.transform(img)
            if isinstance(img, torch.Tensor):
                return tuple(img.shape)  # type: ignore
            return (img.mode == "RGB" and 3 or 1, *img.size)

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self.class_to_idx)

    def get_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from class indices to class names."""
        return {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        img_path = self.files[idx]
        class_name = img_path.parent.name
        class_idx = self.class_to_idx[class_name]

        # Load and transform image
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)

        return img, class_idx

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "CustomImageDataset":
        """Create a dataset instance from a filesystem path."""
        return cls(root=path, **kwargs)

    def _scan_directory(self) -> None:
        """Scan the root directory to find images and build class mapping."""
        # Check if directory exists
        if not self.root.exists():
            raise ValueError(f"Dataset root directory does not exist: {self.root}")

        if not self.root.is_dir():
            raise ValueError(f"Dataset root path is not a directory: {self.root}")

        # Find all subdirectories (class folders)
        try:
            class_dirs = [
                d
                for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        except Exception as e:
            raise ValueError(f"Error scanning dataset directory: {e}")

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root}")

        # Sort class names for deterministic ordering
        class_dirs.sort()

        # Create class mapping
        self.class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

        # Find all valid image files
        for class_dir in class_dirs:
            for ext in self.SUPPORTED_EXTENSIONS:
                self.files.extend(class_dir.glob(f"*{ext}"))

        # Sort files for deterministic ordering
        self.files.sort()

        if not self.files:
            raise ValueError(
                f"No valid images found in {self.root}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        logger.debug("Found classes: %s", ", ".join(self.class_to_idx.keys()))
