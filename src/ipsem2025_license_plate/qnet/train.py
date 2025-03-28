"""Training functionality for hybrid quantum-classical model."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements,import-outside-toplevel,raising-format-tuple,unused-argument

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch
from torch import nn, optim
from qiskit_aer.primitives import SamplerV2
import qiskit_aer

from ..utils.logging_utils import get_logger
from .model import HybridModel
from .test import evaluate_model
from .utils import get_hardware_info, log_training_stats

logger = get_logger(__name__)


def train_model(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=3,
    stats_file: Optional[Path] = None,
):
    """Train the hybrid model using cross entropy loss and Adam optimizer."""
    logger.info("Starting training for %s epochs on %s", epochs, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr = 1e-3

    model.to(device)

    # Track hardware info
    hardware_info = get_hardware_info()
    logger.info("Hardware info: %s", hardware_info)

    training_stats = []
    total_start_time = time.time()
    peak_memory = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = len(train_loader)

        # Training loop
        with torch.set_grad_enabled(True):
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Memory tracking
                if device == "cuda":
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_memory = max(peak_memory, current_memory)

                # Update statistics
                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
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

        # Validation phase
        with torch.no_grad(), torch.inference_mode():
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss / val_total
        val_accuracy = 100.0 * val_correct / val_total

        logger.info(
            "Validation - Loss: %.4f, Accuracy: %.2f%%",
            val_loss,
            val_accuracy,
        )

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
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
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
    model.eval()  # Ensure model is in eval mode after training
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
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    dataset_type: str = "emnist",
    dataset_path: str = "data",
    model_save_path: Optional[str] = None,
    stats_file: Optional[Path] = None,
    log_file: Optional[str] = None,
    run_test: bool = False,
    verbose: bool = False,
    use_gpu_for_qnn: bool = True,
) -> Dict[str, Any]:
    """Train the hybrid quantum-classical model."""
    # Select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Setup GPU-accelerated quantum sampler if requested
    sampler = None
    if use_gpu_for_qnn and torch.cuda.is_available():
        logger.info("Creating GPU-accelerated quantum simulator via qiskit-aer-gpu")
        try:
            # Create simulator with GPU method
            backend = qiskit_aer.AerSimulator()
            
            # Configure GPU in a way compatible with qiskit-aer-gpu
            backend.set_options(device='GPU')
            
            # Create SamplerV2 with the configured backend
            sampler = SamplerV2(backend=backend)
            logger.info("GPU acceleration successfully enabled for quantum simulation")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU quantum simulator: {e}")
            logger.info("Falling back to CPU-based quantum simulation")
            sampler = None
    else:
        if use_gpu_for_qnn and not torch.cuda.is_available():
            logger.warning("GPU requested for QNN but not available")

    from ..datasets.custom import CustomImageDataset
    from ..datasets.emnist import EMNISTDataset
    from ..datasets.mnist import MNISTDataset

    # Load appropriate dataset
    logger.info("Loading %s dataset from %s", dataset_type, dataset_path)
    try:
        if dataset_type.lower() == "emnist":
            dataset = EMNISTDataset(root=dataset_path, train=True, download=True)  # type: ignore
        elif dataset_type.lower() == "mnist":
            dataset = MNISTDataset(root=dataset_path, train=True, download=True)  # type: ignore
        elif dataset_type.lower() == "custom":
            dataset = CustomImageDataset(root=dataset_path)  # type: ignore
        else:
            raise ValueError("Unknown dataset type: %s", dataset_type)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=4,
    )

    # Log the sizes of the datasets
    logger.info(
        "Data loader sizes - Train: %d, Val: %d, Test: %d",
        len(train_loader.dataset),  # type: ignore
        len(val_loader.dataset),  # type: ignore
        len(test_loader.dataset),  # type: ignore
    )

    logger.info(
        "Created data loaders - Train: %d samples, Val: %d samples",  # type: ignore[arg-type]
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),  # type: ignore[arg-type]
    )

    # Create model with GPU sampler if available
    num_classes = dataset.get_num_classes()
    logger.info(
        "Creating hybrid model with %d qubits, %d ansatz repetitions",
        n_qubits,
        ansatz_reps,
    )
    model = HybridModel(
        n_qubits=n_qubits,
        ansatz_reps=ansatz_reps,
        num_classes=num_classes,
        sampler=sampler,
        use_gpu=use_gpu_for_qnn,
    )
    logger.info(
        "Model created with %d parameters", sum(p.numel() for p in model.parameters())
    )

    # Convert stats file to Path if provided
    if stats_file is not None and isinstance(stats_file, str):
        stats_file = Path(stats_file)

    # Train model
    training_stats = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        stats_file=stats_file,
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
