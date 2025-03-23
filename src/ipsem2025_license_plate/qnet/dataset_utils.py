"""
Dataset utilities for EMNIST processing and preparation.
This module handles:
1. Loading and processing the EMNIST dataset
2. Providing the label mapping for EMNIST
3. Creating filtered datasets with proper label mappings
"""

import os

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)


def get_mapping_path():
    """Returns the path to the mapping file, creating it if necessary."""
    # Define directory path
    base_dir = "data/EMNIST/raw"
    os.makedirs(base_dir, exist_ok=True)

    # Define file path
    mapping_path = os.path.join(base_dir, "emnist-bymerge-mapping.txt")

    # Check if mapping file exists, if not create it
    if not os.path.exists(mapping_path):
        logger.info("Creating mapping file at %s", mapping_path)
        with open(mapping_path, "w", encoding="utf-8") as f:
            # These are the mappings provided
            mappings = """0 48
1 49
2 50
3 51
4 52
5 53
6 54
7 55
8 56
9 57
10 65
11 66
12 67
13 68
14 69
15 70
16 71
17 72
18 73
19 74
20 75
21 76
22 77
23 78
24 79
25 80
26 81
27 82
28 83
29 84
30 85
31 86
32 87
33 88
34 89
35 90
36 97
37 98
38 100
39 101
40 102
41 103
42 104
43 110
44 113
45 114
46 116"""
            f.write(mappings)

    return mapping_path


def load_label_to_char_mapping():
    """
    Loads the EMNIST ByMerge mapping file and returns a dictionary
    mapping EMNIST labels to ASCII characters.
    """
    mapping_path = get_mapping_path()
    logger.info("Using mapping file: %s", mapping_path)

    label_to_char = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            emnist_label_str, ascii_code_str = line.split()
            emnist_label = int(emnist_label_str)
            ascii_code = int(ascii_code_str)
            label_to_char[emnist_label] = chr(ascii_code)

    logger.info("Loaded mapping with %s entries.", len(label_to_char))
    logger.debug("Mapping: %s", label_to_char)
    return label_to_char


def build_36class_map(label_to_char_dict):
    """
    Returns a dict: emnist_label -> new_label in [0..35],
    covering digits '0..9' -> [0..9] and letters 'A..Z' -> [10..35].
    Ignores anything else.
    """
    new_map = {}
    for emnist_label, ascii_char in label_to_char_dict.items():
        ascii_char = ascii_char.upper()  # unify letters as uppercase

        # If it's a digit '0'..'9'
        if "0" <= ascii_char <= "9":
            new_label = ord(ascii_char) - ord("0")  # '0'(48)->0, '9'(57)->9
            new_map[emnist_label] = new_label

        # If it's a letter 'A'..'Z'
        elif "A" <= ascii_char <= "Z":
            new_label = ord(ascii_char) - ord("A") + 10  # 'A'->10, 'Z'->35
            new_map[emnist_label] = new_label

    logger.info("Created 36-class map with %s entries (0-9, A-Z)", len(new_map))
    return new_map


class EMNIST36(Dataset):
    """
    Wraps EMNIST ByMerge dataset:
      - Filters out classes not in our 36-class map
      - Remaps old_label to new_label in [0..35]
    """

    def __init__(self, emnist_dataset, label_map_36):
        super().__init__()
        self.emnist_dataset = emnist_dataset
        self.label_map_36 = label_map_36

        logger.info("Filtering data for 36 classes. This may take a moment...")
        self.indices = []
        for i, (_, old_label) in enumerate(self.emnist_dataset):
            if old_label in self.label_map_36:
                self.indices.append(i)

        logger.info(
            "Retaining %s/%s samples in 36-class dataset.",
            len(self.indices),
            len(emnist_dataset),
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_index = self.indices[idx]
        img, old_label = self.emnist_dataset[actual_index]
        new_label = self.label_map_36[old_label]  # map to [0..35]
        return img, new_label


def load_and_prepare_dataset(batch_size=16):
    """
    Prepares the EMNIST dataset for training and testing:
    1. Loads the EMNIST ByMerge dataset
    2. Applies the necessary transforms
    3. Filters to 36 classes (0-9, A-Z)
    4. Creates DataLoaders for training and testing

    Args:
        batch_size: Size of batches to use in DataLoaders

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        num_classes: Number of classes (36)
    """
    # Define transforms
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),  # invert colors (optional)
        ]
    )

    # Load the EMNIST dataset
    logger.info("Loading EMNIST ByMerge dataset...")
    try:
        train_emnist = torchvision.datasets.EMNIST(
            root="data",
            split="bymerge",
            train=True,
            download=True,
            transform=transform_pipeline,
        )
        test_emnist = torchvision.datasets.EMNIST(
            root="data",
            split="bymerge",
            train=False,
            download=True,
            transform=transform_pipeline,
        )
        logger.info(
            "EMNIST dataset loaded successfully: %s training, %s testing samples",
            len(train_emnist),
            len(test_emnist),
        )
    except Exception as e:
        logger.error("Failed to download dataset: %s", e)
        logger.info("Using fallback method to load dataset...")
        # If there's an alternative way to load the dataset, it would go here
        raise

    # Get the label mapping and build 36-class map
    label_to_char = load_label_to_char_mapping()
    mapping_36 = build_36class_map(label_to_char)

    # Create filtered datasets
    train_dataset_36 = EMNIST36(train_emnist, mapping_36)
    test_dataset_36 = EMNIST36(test_emnist, mapping_36)

    # Create DataLoaders
    logger.info("Creating DataLoaders with batch_size=%s", batch_size)
    train_loader = DataLoader(train_dataset_36, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_36, batch_size=batch_size, shuffle=False)

    logger.info(
        "DataLoaders created: %s training batches, %s testing batches",
        len(train_loader),
        len(test_loader),
    )

    return train_loader, test_loader, 36  # 36 classes
