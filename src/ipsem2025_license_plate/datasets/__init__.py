"""Dataset package for IPSEM 2025 License Plate recognition."""

from .base import BaseDataset
from .custom import CustomImageDataset
from .emnist import EMNISTDataset
from .plate_characters import PlateCharactersDataset

__all__ = [
    "BaseDataset",
    "EMNISTDataset",
    "CustomImageDataset",
    "PlateCharactersDataset",
]
