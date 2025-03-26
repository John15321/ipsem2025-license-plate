# pylint: disable=broad-exception-caught,unspecified-encoding,unused-import
"""Utility functions for quantum-classical hybrid model."""

import csv
import fcntl
import os
import pickle
import platform
from pathlib import Path
from typing import Any, Dict, Tuple

import psutil
import torch
from torchvision import transforms

from ..utils.logging_utils import get_logger
from .model import HybridModel

logger = get_logger(__name__)


def get_hardware_info() -> Dict[str, str]:
    """Collect system hardware information."""
    return {
        "cpu_model": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "total_memory": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
        "cpu_count": str(psutil.cpu_count(logical=False)),
        "cpu_threads": str(psutil.cpu_count(logical=True)),
    }


def log_training_stats(stats_file: Path, stats: dict):
    """Log training statistics to CSV file with file locking for safety."""
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(stats_file, mode="a", newline="") as f:
            # Get an exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
            if stats_file.stat().st_size == 0:  # Only write header for empty files
                writer.writeheader()
            writer.writerow(stats)
            f.flush()  # Force write to disk
            os.fsync(f.fileno())  # Ensure it's written to disk

            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error("Failed to write training stats: %s", e)


def save_model(model: HybridModel, save_path: Path, metadata: dict):
    """Save model and its metadata to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model state and metadata
    model_data = {"state_dict": model.state_dict(), "metadata": metadata}

    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("Model saved to %s", save_path)


def load_model(load_path: Path) -> Tuple[HybridModel, dict]:
    """Load model and its metadata from disk."""
    with open(load_path, "rb") as f:
        model_data = pickle.load(f)

    metadata = model_data["metadata"]

    model = HybridModel(
        n_qubits=metadata["n_qubits"],
        ansatz_reps=metadata["ansatz_reps"],
        num_classes=metadata["num_classes"],
    )
    model.load_state_dict(model_data["state_dict"])

    logger.info("Model loaded from %s", load_path)
    return model, metadata


def get_default_transform() -> transforms.Compose:
    """Return the default transformation pipeline for datasets."""
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
