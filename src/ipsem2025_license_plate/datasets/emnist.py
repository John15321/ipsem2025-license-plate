"""EMNIST dataset implementation."""

import os
from typing import Dict, Tuple, Optional, Any

import torch
import torchvision
from torchvision import transforms

from .base import BaseDataset
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class EMNISTDataset(BaseDataset):
    """EMNIST dataset filtered to 36 classes (0-9, A-Z)."""
    
    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True
    ):
        """Initialize the EMNIST dataset.
        
        Args:
            root: Root directory for dataset storage
            train: Whether to load training or test set
            transform: Optional transform to apply to images
            download: Whether to download the dataset if not found
        """
        super().__init__()
        self.root = root  # Store root path for use in other methods
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x),  # invert colors
            ])
        
        # Load the EMNIST dataset
        self.emnist = torchvision.datasets.EMNIST(
            root=root,
            split="bymerge",
            train=train,
            download=download,
            transform=transform
        )
        
        # Load and create label mappings
        self.label_to_char = self._load_label_mapping()
        self.label_map_36 = self._build_36class_map()
        
        # Filter indices to only include our 36 classes
        self.indices = [
            i for i, (_, label) in enumerate(self.emnist)
            if label in self.label_map_36
        ]
        
        logger.info(
            "Initialized EMNISTDataset with %d/%d samples (36-class filtered)",
            len(self.indices),
            len(self.emnist)
        )

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of images in the dataset."""
        # EMNIST images are grayscale 64x64 after our transform
        return (1, 64, 64)

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return 36  # 10 digits + 26 letters

    def get_class_mapping(self) -> Dict[int, str]:
        """Get the mapping from class indices to class names."""
        mapping = {}
        for emnist_label, new_label in self.label_map_36.items():
            char = self.label_to_char[emnist_label]
            mapping[new_label] = char
        return mapping

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        actual_index = self.indices[idx]
        img, old_label = self.emnist[actual_index]
        new_label = self.label_map_36[old_label]  # map to [0..35]
        return img, new_label

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> 'EMNISTDataset':
        """Create a dataset instance from a filesystem path."""
        return cls(root=path, **kwargs)

    def _load_label_mapping(self) -> Dict[int, str]:
        """Load the EMNIST ByMerge mapping file."""
        # Define directory path using the instance root path
        base_dir = os.path.join(self.root, "EMNIST", "raw")
        os.makedirs(base_dir, exist_ok=True)
        
        # Define file path
        mapping_path = os.path.join(base_dir, "emnist-bymerge-mapping.txt")
        
        # Create mapping file if it doesn't exist
        if not os.path.exists(mapping_path):
            logger.info("Creating mapping file at %s", mapping_path)
            mappings = "\n".join([
                f"{i} {ord('0') + i}" for i in range(10)  # 0-9
            ] + [
                f"{i + 10} {ord('A') + i}" for i in range(26)  # A-Z
            ])
            with open(mapping_path, "w", encoding="utf-8") as f:
                f.write(mappings)
        
        # Load mappings
        label_to_char = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                label_str, ascii_code_str = line.split()
                label = int(label_str)
                ascii_code = int(ascii_code_str)
                label_to_char[label] = chr(ascii_code)
        
        logger.info("Loaded EMNIST mapping with %d entries", len(label_to_char))
        return label_to_char

    def _build_36class_map(self) -> Dict[int, int]:
        """Build mapping from EMNIST labels to 0-35 range."""
        new_map = {}
        for emnist_label, ascii_char in self.label_to_char.items():
            ascii_char = ascii_char.upper()  # unify letters as uppercase
            
            # Map digits 0-9
            if "0" <= ascii_char <= "9":
                new_label = ord(ascii_char) - ord("0")
                new_map[emnist_label] = new_label
            
            # Map letters A-Z to 10-35
            elif "A" <= ascii_char <= "Z":
                new_label = ord(ascii_char) - ord("A") + 10
                new_map[emnist_label] = new_label
        
        logger.info("Created 36-class map: 0-9, A-Z")
        return new_map