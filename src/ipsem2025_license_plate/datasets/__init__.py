"""Dataset package for IPSEM 2025 License Plate recognition."""

from .base import BaseDataset
from .emnist import EMNISTDataset
from .custom import CustomImageDataset

__all__ = [
    'BaseDataset',
    'EMNISTDataset',
    'CustomImageDataset',
]