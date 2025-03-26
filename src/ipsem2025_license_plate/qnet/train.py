"""Training functionality for hybrid quantum-classical model."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.logging_utils import get_logger
from .model import HybridModel
from .test import evaluate_model
from .utils import get_hardware_info, log_training_stats

logger = get_logger(__name__)


def train_model(
    model, train_loader, device="cpu", epochs=3, stats_file: Optional[Path] = None
):
    """Train the hybrid model using cross entropy loss and Adam optimizer."""
    logger.info("Starting training for %s epochs on %s", epochs, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr = 1e-3

    model.to(device)
    model.train()

    # Track hardware info
    hardware_info = get_hardware_info()
    logger.info("Hardware info: %s", hardware_info)

    training_stats = []
    total_start_time = time.time()
    peak_memory = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Memory tracking
            if device == "cuda":
                current_memory = torch.cuda.memory_allocated() / 1024**2
                peak_memory = max(peak_memory, current_memory)

            # Update statistics
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

            # Log progress every 10% of batches
            if batch_idx % max(1, batch_count // 10) == 0:
                current_loss = total_loss / total_samples
                current_acc = 100.0 * total_correct / total_samples
                logger.info(
                    "Epoch %d [%d/%d] Loss: %.4f Acc: %.1f%%",
                    epoch + 1,
                    batch_idx,
                    batch_count,
                    current_loss,
                    current_acc,
                )

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        # Log epoch summary
        logger.info(
            "Epoch %d/%d completed in %.2fs - Loss: %.4f, Accuracy: %.2f%%, Memory: %.1fMB",
            epoch + 1,
            epochs,
            epoch_time,
            avg_loss,
            accuracy,
            peak_memory,
        )

        # Collect and immediately save epoch statistics
        epoch_stats = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples_processed": total_samples,
            "epoch_time": epoch_time,
            "total_time": time.time() - total_start_time,
            "learning_rate": lr,
            "cpu_memory_mb": psutil.Process().memory_info().rss / 1024**2,
            "gpu_memory_mb": (
                torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
            ),
            "batch_size": train_loader.batch_size,
            **hardware_info,
        }

        training_stats.append(epoch_stats)

        # Immediately write epoch stats if stats_file is provided
        if stats_file is not None:
            log_training_stats(stats_file, epoch_stats)

    total_time = time.time() - total_start_time
    logger.info(
        "Training complete! Total time: %.2fs, Final accuracy: %.2f%%",
        total_time,
        accuracy,
    )
    return training_stats


def train_hybrid_model(
    n_qubits: int = 2,
    ansatz_reps: int = 1,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    dataset_type: str = "emnist",
    dataset_path: str = "data",
    model_save_path: Optional[str] = None,
    stats_file: Optional[Path] = None,
    log_file: Optional[str] = None,
    run_test: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train the hybrid quantum-classical model."""
    # Select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    from ..datasets.custom import CustomImageDataset
    from ..datasets.emnist import EMNISTDataset
    from ..datasets.mnist import MNISTDataset

    # Load appropriate dataset
    logger.info("Loading %s dataset from %s", dataset_type, dataset_path)
    if dataset_type.lower() == "emnist":
        dataset = EMNISTDataset(root=dataset_path, train=True, download=True)
    elif dataset_type.lower() == "mnist":
        dataset = MNISTDataset(root=dataset_path, train=True, download=True)
    elif dataset_type.lower() == "custom":
        dataset = CustomImageDataset(root=dataset_path)
    else:
        raise ValueError("Unknown dataset type: %s", dataset_type)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        batch_size=batch_size, train_ratio=0.7, val_ratio=0.15, num_workers=4
    )
    logger.info("Created data loaders - Train: %d samples", len(train_loader.dataset))

    # Create model
    num_classes = dataset.get_num_classes()
    logger.info(
        "Creating hybrid model with %d qubits, %d ansatz repetitions",
        n_qubits,
        ansatz_reps,
    )
    model = HybridModel(
        n_qubits=n_qubits, ansatz_reps=ansatz_reps, num_classes=num_classes
    )
    logger.info(
        "Model created with %d parameters", sum(p.numel() for p in model.parameters())
    )

    # Convert stats file to Path if provided
    if stats_file is not None:
        stats_file = Path(stats_file)

    # Train model
    training_stats = train_model(
        model, train_loader, device=device, epochs=epochs, stats_file=stats_file
    )

    # Run test if requested
    test_metrics = None
    if run_test:
        logger.info("Running model evaluation on test set...")
        test_metrics = evaluate_model(model, test_loader, device)

    # Save model if path provided
    if model_save_path:
        from .utils import save_model

        save_path = Path(model_save_path)
        metadata = {
            "n_qubits": n_qubits,
            "ansatz_reps": ansatz_reps,
            "num_classes": num_classes,
            "dataset_type": dataset_type,
            "test_accuracy": test_metrics["test_accuracy"] if test_metrics else None,
            "training_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "timestamp": datetime.now().isoformat(),
        }
        save_model(model, save_path, metadata)

    return {
        "model": model,
        "final_stats": training_stats[-1] if training_stats else None,
        "training_history": training_stats,
        "test_metrics": test_metrics,
        "hardware_info": get_hardware_info(),
    }
