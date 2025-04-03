import os
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision
from PIL import Image, ImageOps
from tqdm import tqdm  # For progress bar
from ..qnet.utils import get_default_transform
from ..utils.logging_utils import get_logger
from .base import BaseDataset

logger = get_logger(__name__)

class PlateCharactersDataset(BaseDataset):
    """Dataset for character extraction from license plates."""

    def __init__(
        self,
        data_dir: str = "data",
        transform: Optional[torchvision.transforms.Compose] = None,
        target_height: int = 128,
    ):
        """Initialize the PlateCharacters dataset.

        Args:
            data_dir: Root directory for dataset storage
            transform: Optional transform to apply to images
            target_height: Target height to which images will be resized
        """
        super().__init__()
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        self.image_paths = []
        self.labels = []
        self.target_height = target_height

        if transform is None:
            transform = get_default_transform()
        self.transform = transform

        # Load image paths and labels with a progress bar
        image_files = os.listdir(self.image_dir)
        for img_file in tqdm(image_files, desc="Loading images and labels", unit="image"):
            img_path = os.path.join(self.image_dir, img_file)
            label_path = os.path.join(self.label_dir, f"{os.path.splitext(img_file)[0]}.txt")
            
            if os.path.isfile(img_path) and os.path.isfile(label_path):
                self.image_paths.append(img_path)
                with open(label_path, "r") as label_file:
                    self.labels.append(label_file.read().strip())

        logger.info(
            "Initialized PlateCharactersDataset with %d samples",
            len(self.image_paths),
        )

    def _resize_and_pad(self, image: Image.Image) -> Image.Image:
        """Resize the image to the target height while maintaining aspect ratio, then pad to square."""
        aspect_ratio = image.width / image.height
        new_width = int(self.target_height * aspect_ratio)
        image = image.resize((new_width, self.target_height), Image.Resampling.LANCZOS)

        # Pad to square
        delta_width = self.target_height - new_width
        padding = (delta_width // 2, 0, delta_width - delta_width // 2, 0)
        image = ImageOps.expand(image, padding, fill=255)

        return image

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of images in the dataset."""
        return (1, self.target_height, self.target_height)  # Square images after padding

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(set(self.labels))

    def get_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from class indices to class names."""
        return {i: str(i) for i in range(len(set(self.labels)))}

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")  # Convert to grayscale
        image = self._resize_and_pad(image)  # Resize and pad to square
        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "PlateCharactersDataset":
        """Create a dataset instance from a filesystem path."""
        transform = kwargs.get("transform", get_default_transform())
        target_height = kwargs.get("target_height", 128)
        return cls(data_dir=path, transform=transform, target_height=target_height)

    @staticmethod
    def exists_at_path(path: str) -> bool:
        """Check if PlateCharacters dataset exists at the given path.

        Args:
            path: Path to check for dataset

        Returns:
            True if dataset exists, False otherwise
        """
        required_directories = [
            "images",  # images directory should exist
            "labels",  # labels directory should exist
        ]
        
        # Check if the required directories exist
        return all(os.path.exists(os.path.join(path, dir)) for dir in required_directories)
