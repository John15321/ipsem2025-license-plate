"""EMNIST dataset implementation."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

import multiprocessing
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils.logging_utils import get_logger
from .base import BaseDataset

logger = get_logger(__name__)


class EMNISTDataset(BaseDataset):
    """EMNIST dataset filtered to 36 classes (0-9, A-Z)."""

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        lazy_load: bool = False,
        preload_to_memory: bool = False,  # Preload entire dataset to memory
    ):
        """Initialize the EMNIST dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to load training or test set
            transform: Optional transform to apply to images
            download: Whether to download the dataset if not found
            lazy_load: Whether to delay loading the dataset until needed
            preload_to_memory: Whether to load the entire dataset into memory
        """
        super().__init__()
        self.root = root  # Store root path for use in other methods
        self.train = train
        self.download = download
        self._emnist = None  # Will hold the actual dataset when loaded
        self.preload_to_memory = preload_to_memory
        self.preloaded_data = None  # Will hold preloaded data

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: 1.0 - x),  # invert colors
                ]
            )
        self.transform = transform

        # Load label mappings (these are small and fast to load)
        self.label_to_char = self._load_label_mapping()
        self.label_map_36 = self._build_36class_map()

        # Initialize indices list (will be populated on first load)
        self.indices: List[int] = []

        # Load dataset if not using lazy loading
        if not lazy_load:
            self._load_dataset()
            logger.info(
                "Initialized EMNISTDataset with %d/%d samples (36-class filtered)",
                len(self.indices),
                len(self._emnist),  # type: ignore
            )

            # Preload data if requested
            if preload_to_memory:
                self._preload_data_to_memory()

    def _preload_data_to_memory(self):
        """Preload all dataset samples into memory for faster access."""
        if self.preloaded_data is not None:
            return  # Already preloaded

        logger.info("Preloading entire dataset into memory...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        self.preloaded_data = []
        for idx in range(len(self.indices)):
            actual_index = self.indices[idx]
            img, old_label = self.emnist[actual_index]
            new_label = self.label_map_36[old_label]
            self.preloaded_data.append((img, new_label))
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

        memory_usage = sum(
            img.element_size() * img.nelement() for img, _ in self.preloaded_data
        ) / (1024 * 1024)
        logger.info(
            f"Dataset preloaded into memory in {elapsed_time:.2f} seconds. Memory usage: {memory_usage:.2f} MB"
        )

    @property
    def emnist(self):
        """Lazy-load the EMNIST dataset on first access."""
        if self._emnist is None:
            self._load_dataset()
        return self._emnist

    def _load_dataset(self):
        """Load the actual EMNIST dataset and filter indices."""
        if self._emnist is not None:
            return  # Already loaded

        logger.info("Loading EMNIST dataset from %s...", self.root)
        # Load the EMNIST dataset
        self._emnist = torchvision.datasets.EMNIST(
            root=self.root,
            split="bymerge",
            train=self.train,
            download=self.download,
            transform=self.transform,
        )

        # Ensure indices are correctly filtered and synchronized
        if not self.indices:
            self.indices = [
                i
                for i, (_, label) in enumerate(self._emnist)
                if label in self.label_map_36
            ]

        # Log the number of filtered indices
        logger.info(
            "Filtered dataset to %d samples out of %d total samples",
            len(self.indices),
            len(self._emnist),
        )

        logger.info(
            "EMNIST dataset loaded with %d/%d samples (36-class filtered)",
            len(self.indices),
            len(self._emnist),
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
        if not self.indices:
            self._load_dataset()
        if self._emnist is None:
            self._load_dataset()
        if self._emnist is None:
            raise ValueError(
                "The EMNIST dataset has not been loaded. Ensure it is "
                "initialized before accessing its length."
            )
        # Return the length of filtered indices instead of the whole dataset
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        # If data is preloaded, fetch directly from memory
        if self.preloaded_data is not None:
            if idx < 0 or idx >= len(self.preloaded_data):
                raise IndexError(
                    f"Index {idx} out of range for dataset with {len(self.preloaded_data)} samples"
                )
            return self.preloaded_data[idx]

        # Otherwise load from disk
        if idx < 0 or idx >= len(self.indices):
            logger.error(
                "Index %d is out of range for dataset with %d samples. This may indicate a mismatch between the dataset and DataLoader.",  # pylint: disable=line-too-long
                idx,
                len(self.indices),
            )
            raise IndexError(
                f"Index {idx} is out of range for dataset with {len(self.indices)} samples."
            )

        actual_index = self.indices[idx]
        img, old_label = self.emnist[actual_index]
        new_label = self.label_map_36[old_label]  # map to [0..35]
        return img, new_label

    def create_data_loaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create optimized data loaders for training, validation and testing.

        Args:
            batch_size: Batch size for all data loaders
            train_ratio: Portion of dataset to use for training
            val_ratio: Portion of dataset to use for validation
            num_workers: Number of worker processes for data loading (defaults to CPU count)
            pin_memory: Whether to use pinned memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Whether to keep worker processes alive between epochs

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Automatically use all available cores if not specified
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)  # Use at most 8 workers
            logger.info(
                f"Using {num_workers} dataloader workers (detected {multiprocessing.cpu_count()} CPU cores)"
            )

        # Prepare data splits
        dataset_size = len(self)
        if dataset_size == 0:
            raise ValueError("Dataset is empty, cannot create data loaders")

        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(
                f"Invalid split sizes: train={train_size}, val={val_size}, test={test_size}. "
                f"Check your train_ratio ({train_ratio}) and val_ratio ({val_ratio})."
            )

        # Create random splits
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self, [train_size, val_size, test_size]
        )

        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True,  # Drop last batch for more consistent training
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers and num_workers > 0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers and num_workers > 0,
        )

        logger.info(
            f"Created data loaders with batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}, "
            f"sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "EMNISTDataset":
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
            mappings = "\n".join(
                [f"{i} {ord('0') + i}" for i in range(10)]  # 0-9
                + [f"{i + 10} {ord('A') + i}" for i in range(26)]  # A-Z
            )
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

    @staticmethod
    def exists_at_path(path: str) -> bool:
        """Check if an EMNIST dataset exists at the given path.

        This method checks if the dataset files exist without actually loading
        the dataset, which makes it much faster for validation.

        Args:
            path: Path to check for dataset files

        Returns:
            True if dataset files exist, False otherwise
        """
        # Check for the processed files that indicate a downloaded dataset
        processed_folder = os.path.join(path, "EMNIST", "processed")
        return (
            os.path.exists(processed_folder) and len(os.listdir(processed_folder)) > 0
        )
