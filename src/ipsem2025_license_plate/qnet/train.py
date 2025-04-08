"""Training functionality for hybrid quantum-classical model."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements,import-outside-toplevel,raising-format-tuple,unused-argument

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil
import qiskit_aer
import torch
from qiskit_aer.primitives import SamplerV2
from torch import nn, optim
from torch.optim import LBFGS

from ..utils.logging_utils import get_logger
from .model import HybridModel
from .test import evaluate_model
from .utils import get_hardware_info, log_training_stats

logger = get_logger(__name__)


def train_model(
    model,
    train_loader,
    val_loader,
    device: Optional[torch.device] = None,
    epochs=3,
    stats_file: Optional[Path] = None,
    learning_rate: float = 1e-3,
    run_dir: Optional[Path] = None,  # Added parameter for the run directory
):
    """Train the hybrid model using cross entropy loss and Adam optimizer.

    Args:
        model: The HybridModel to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training (torch.device object)
        epochs: Number of training epochs
        stats_file: Path to save training statistics
        learning_rate: Learning rate for optimizer
        run_dir: Directory to save intermediate models

    Returns:
        List of training statistics dictionaries, one per epoch
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting training for %s epochs on %s", epochs, device)

    # Log detailed GPU information if using CUDA
    if device.type == "cuda":
        logger.info(f"Training on CUDA Device: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"Current GPU memory usage: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
        )
        logger.info(
            f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB"
        )
        logger.info(
            f"Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB"
        )

    criterion = nn.MSELoss().to(device)
    logger.info(f"Loss function moved to device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = LBFGS(model.parameters())
    logger.info(f"Optimizer created with learning rate: {learning_rate}")

    # Ensure model is on the correct device
    model.to(device)

    # Log device placement for all model parameters
    logger.info("Verifying model parameter device placement:")
    for name, param in model.named_parameters():
        logger.info(f"  - Parameter '{name}' on {param.device}")

    # Track hardware info
    hardware_info = get_hardware_info()
    logger.info("Hardware info: %s", hardware_info)

    training_stats = []
    total_start_time = time.time()
    peak_memory = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)

        if device.type == "cuda":
            logger.info(
                f"GPU memory before epoch: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
            )
            # Clear GPU cache before each epoch to minimize memory fragmentation
            torch.cuda.empty_cache()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = len(train_loader)

        # Training loop
        with torch.set_grad_enabled(True):
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader, 1):
                # Move data to device and use non_blocking for potential speed improvement
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Log device placement for first batch
                if batch_idx == 1 and epoch == 0:
                    logger.info(
                        f"Batch data device: images={images.device}, labels={labels.device}"
                    )

                optimizer.zero_grad(set_to_none=True)  # More efficient version
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Memory tracking
                if device.type == "cuda":
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_memory = max(peak_memory, current_memory)

                    # Log memory usage every 10% of batches
                    if batch_idx % max(1, batch_count // 10) == 0:
                        logger.info(
                            f"GPU memory at batch {batch_idx}: {current_memory:.2f} MB"
                        )

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
                # Move data to device and use non_blocking for potential speed improvement
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

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

        # Log GPU memory stats after validation
        if device.type == "cuda":
            logger.info(
                f"GPU memory after validation: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
            )
            logger.info(f"Peak GPU memory this epoch: {peak_memory:.2f} MB")

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

        # Save intermediate model after each epoch if run_dir is provided
        if run_dir is not None:
            epoch_model_path = run_dir / f"model_epoch_{epoch+1:03d}.pt"
            try:
                torch.save(model.state_dict(), epoch_model_path)
                logger.info(
                    f"Saved intermediate model after epoch {epoch+1} to {epoch_model_path}"
                )

                # Also save the current epoch stats in its own file
                epoch_stats_path = run_dir / f"stats_epoch_{epoch+1:03d}.json"
                with open(epoch_stats_path, "w") as f:
                    import json

                    # Convert timestamp to string for JSON serialization
                    epoch_stats_json = epoch_stats.copy()
                    epoch_stats_json["timestamp"] = str(epoch_stats_json["timestamp"])
                    json.dump(epoch_stats_json, f, indent=2)
                logger.info(f"Saved epoch stats to {epoch_stats_path}")
            except Exception as e:
                logger.error(
                    f"Failed to save intermediate model after epoch {epoch+1}: {e}"
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
            "learning_rate": learning_rate,
            "cpu_memory_mb": psutil.Process().memory_info().rss / 1024**2,
            "gpu_memory_mb": (
                torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0
            ),
            "gpu_peak_memory_mb": peak_memory,
            "batch_size": train_loader.batch_size,
            "device": str(device),  # Include device info in stats
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

    # Log final GPU memory usage
    if device.type == "cuda":
        logger.info(
            f"Final GPU memory usage: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
        )
        logger.info(
            f"Peak GPU memory across all epochs: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB"
        )

    return training_stats


def train_hybrid_model(
    n_qubits: int = 6,
    ansatz_reps: int = 2,
    feature_map_reps: int = 1,
    epochs: int = 3,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    learning_rate: float = 1e-3,
    device: Optional[
        Union[str, torch.device]
    ] = None,  # Accept both string and device objects
    dataset_type: str = "emnist",
    dataset_path: str = "data",
    model_save_path: Optional[str] = None,
    stats_file: Optional[Path] = None,
    run_dir: Optional[
        Path
    ] = None,  # Now accepting run_dir instead of log_file and output_dir
    run_test: bool = False,
    verbose: bool = False,
    use_gpu_for_qnn: bool = True,
    save_intermediate: bool = True,  # Whether to save intermediate models
    preload_data: bool = False,  # Whether to preload dataset into memory
    num_workers: Optional[int] = None,  # Number of dataloader workers
) -> Dict[str, Any]:
    """Train the hybrid quantum-classical model.

    Args:
        n_qubits: Number of qubits in the quantum circuit (default: 6).
        ansatz_reps: Number of repetitions in the RealAmplitudes ansatz (default: 2).
        feature_map_reps: Number of repetitions in the feature map (default: 1).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        train_ratio: Ratio of data used for training.
        val_ratio: Ratio of data used for validation.
        learning_rate: Learning rate for optimizer.
        device: Device to use for classical computation (torch.device object or string 'cpu'/'cuda').
        dataset_type: Type of dataset ('emnist', 'mnist', or 'custom').
        dataset_path: Path to dataset.
        model_save_path: Path to save trained model.
        stats_file: Path to save training statistics.
        run_dir: Directory for all run outputs.
        run_test: Whether to run evaluation after training.
        verbose: Enable verbose output.
        use_gpu_for_qnn: Whether to use GPU acceleration for quantum circuit simulation.
        save_intermediate: Whether to save intermediate models after each epoch.
        preload_data: Whether to preload dataset into memory.
        num_workers: Number of dataloader workers.

    Returns:
        A dictionary containing the trained model, training stats, and test metrics.
    """
    # We no longer need to create the run directory as it's now passed in
    # Ensure run directory exists
    if run_dir is None:
        logger.warning("No run directory provided, using current directory")
        run_dir = Path(".")

    logger.info("===========================================================")
    logger.info("Starting hybrid quantum-classical neural network training")
    logger.info(f"Run output directory: {run_dir}")
    logger.info("===========================================================")

    # Convert string device specification to torch.device if needed
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Log CUDA information if available
    if device.type == "cuda":
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"CUDA Device Capability: {torch.cuda.get_device_capability(device)}"
        )
        logger.info(f"CUDA Current Device: {torch.cuda.current_device()}")
        logger.info(
            f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB total"
        )

    logger.info("Configuration:")
    logger.info(
        "  - Quantum configuration: %d qubits, %d ansatz repetitions, %d feature map repetitions",
        n_qubits,
        ansatz_reps,
        feature_map_reps,
    )
    logger.info("  - Training parameters: %d epochs, batch size %d", epochs, batch_size)
    logger.info("  - Dataset: %s from %s", dataset_type, dataset_path)
    logger.info("  - Device: %s", device)
    logger.info(
        "  - Data split: %.1f%% train, %.1f%% validation, %.1f%% test",
        train_ratio * 100,
        val_ratio * 100,
        (1 - train_ratio - val_ratio) * 100,
    )

    # Log hardware information
    hw_info = get_hardware_info()
    logger.info("Hardware information:")
    logger.info(
        "  - CPU: %s with %s cores (%s threads)",
        hw_info.get("cpu_model", "Unknown"),
        hw_info.get("cpu_count", "Unknown"),
        hw_info.get("cpu_threads", "Unknown"),
    )
    logger.info("  - Memory: %s", hw_info.get("total_memory", "Unknown"))

    if device.type == "cuda":
        logger.info("  - GPU: %s", hw_info.get("gpu_model", "Unknown"))
        logger.info("  - GPU Memory: %s", hw_info.get("gpu_memory", "Unknown"))

    # Setup GPU-accelerated quantum sampler if requested
    logger.info("Initializing quantum simulation backend...")
    sampler = None
    if use_gpu_for_qnn and device.type == "cuda":
        logger.info("Creating GPU-accelerated quantum simulator via qiskit-aer-gpu")
        try:
            # Configure GPU in the options dictionary, not as a direct backend parameter
            backend_options = {"method": "statevector"}
            run_options = {"device": "GPU"}

            # Create SamplerV2 with the correct options structure
            sampler = SamplerV2(
                options={"backend_options": backend_options, "run_options": run_options}
            )
            logger.info("GPU acceleration successfully enabled for quantum simulation")

            # Try to verify Aer backend is using GPU
            try:
                from qiskit_aer import AerSimulator

                sim = AerSimulator(method="statevector", device="GPU")
                logger.info(f"AerSimulator GPU check: {sim.available_devices()}")
                if "GPU" in sim.available_devices():
                    logger.info("GPU confirmed available for AerSimulator")
                else:
                    logger.warning("GPU not found in AerSimulator available devices")
            except Exception as e:
                logger.warning(
                    f"Could not verify GPU availability for AerSimulator: {e}"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize GPU quantum simulator: {e}")
            logger.info("Falling back to CPU-based quantum simulation")
            sampler = None
    else:
        if use_gpu_for_qnn and device.type != "cuda":
            logger.warning("GPU requested for QNN but not available")
            logger.info("Using CPU-based quantum simulation")

    logger.info("Loading dataset modules...")
    from ..datasets.custom import CustomImageDataset
    from ..datasets.emnist import EMNISTDataset
    from ..datasets.mnist import MNISTDataset

    # Load appropriate dataset
    logger.info("Loading %s dataset from %s...", dataset_type, dataset_path)
    try:
        if dataset_type.lower() == "emnist":
            logger.info("Initializing EMNIST dataset (this may take a moment)...")
            dataset = EMNISTDataset(
                root=dataset_path,
                train=True,
                download=True,
                preload_to_memory=preload_data,  # Use new preload parameter
            )
            logger.info("EMNIST dataset loaded successfully")
        elif dataset_type.lower() == "mnist":
            logger.info("Initializing MNIST dataset...")
            dataset = MNISTDataset(
                root=dataset_path,
                train=True,
                download=True,
                preload_to_memory=preload_data,  # Use new preload parameter if it exists
            )
            logger.info("MNIST dataset loaded successfully")
        elif dataset_type.lower() == "custom":
            logger.info("Loading custom image dataset...")
            dataset = CustomImageDataset(
                root=dataset_path,
                preload_to_memory=preload_data,  # Use new preload parameter if it exists
            )
            logger.info("Custom dataset loaded successfully")
        else:
            raise ValueError("Unknown dataset type: %s", dataset_type)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise

    # Create data loaders with pin_memory for faster GPU transfer
    logger.info("Creating data loaders with batch size %d...", batch_size)
    use_pin_memory = device.type == "cuda"
    logger.info(f"Using pin_memory={use_pin_memory} for data loaders")

    # Use the new data loader creation method with num_workers
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=num_workers,  # Pass the num_workers parameter
        pin_memory=use_pin_memory,
    )

    # Log the sizes of the datasets
    logger.info(
        "Data split complete - Train: %d samples, Val: %d samples, Test: %d samples",
        len(train_loader.dataset),  # type: ignore
        len(val_loader.dataset),  # type: ignore
        len(test_loader.dataset),  # type: ignore
    )

    # Log dataset class distribution
    logger.info("Dataset class distribution:")
    class_mapping = dataset.get_class_mapping()
    for class_idx in sorted(list(class_mapping.keys()))[:10]:  # Show first 10 classes
        logger.info("  - Class %d: %s", class_idx, class_mapping[class_idx])
    if len(class_mapping) > 10:
        logger.info("  - ...and %d more classes", len(class_mapping) - 10)

    # Create model with GPU sampler if available
    num_classes = dataset.get_num_classes()
    logger.info(
        "Creating hybrid model with %d qubits, %d feature map reps, %d ansatz repetitions, %d output classes",
        n_qubits,
        feature_map_reps,
        ansatz_reps,
        num_classes,
    )
    model = HybridModel(
        n_qubits=n_qubits,
        ansatz_reps=ansatz_reps,
        feature_map_reps=feature_map_reps,
        num_classes=num_classes,
        use_gpu=use_gpu_for_qnn and device.type == "cuda",
        device=device,  # Pass the device to the model
    )
    model_info = model.get_model_info()
    logger.info(
        "Model created with %d parameters (%d classical, %d quantum)",
        model_info["total_params"],
        model_info["classical_params"],
        model_info["quantum_params"],
    )
    logger.info("Quantum circuit depth: %d", model_info["circuit_depth"])
    logger.info(
        "Using SamplerQNN on %s",
        "GPU" if model_info["using_gpu_quantum"] else "CPU",
    )

    # Explicitly move model to device again and verify all parameters are on correct device
    model.to(device)
    logger.info("Verifying all model parameters are on correct device:")
    all_on_device = True
    for name, param in model.named_parameters():
        if param.device != device:
            logger.warning(f"Parameter {name} is on {param.device}, not {device}")
            all_on_device = False

    if all_on_device:
        logger.info(f"All model parameters are correctly on {device}")
    else:
        logger.warning("Some parameters are not on the requested device")

    # Convert stats file to Path if provided
    if stats_file is not None and isinstance(stats_file, str):
        stats_file = Path(stats_file)
        logger.info("Will save training statistics to %s", stats_file)

    if model_save_path:
        logger.info("Will save trained model to %s", model_save_path)

    # Update stats_file path if it's not provided but we have a run directory
    if stats_file is None:
        stats_file = run_dir / "training_stats.csv"
        logger.info(f"Will save training statistics to {stats_file}")

    # Update model_save_path if it's not provided but we have a run directory
    if model_save_path is None:
        model_save_path = str(run_dir / "model_final.pt")
        logger.info(f"Will save final model to {model_save_path}")

    logger.info("Starting training...")

    # Train model
    training_stats = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        stats_file=stats_file,
        learning_rate=learning_rate,
        run_dir=run_dir,  # Pass the run directory to enable intermediate saves
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
            "feature_map_reps": feature_map_reps,
            "num_classes": num_classes,
            "dataset_type": dataset_type,
            "test_accuracy": test_metrics["test_accuracy"] if test_metrics else None,
            "training_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "run_name": run_name,
        }
        save_model(model, save_path, metadata)

        # Also save the metadata separately as JSON
        if run_dir is not None:
            try:
                import json

                with open(run_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved run metadata to {run_dir / 'metadata.json'}")
            except Exception as e:
                logger.warning(f"Failed to save metadata JSON: {e}")

    # Log final GPU memory stats if using CUDA
    if device.type == "cuda":
        logger.info("Final GPU Memory Statistics:")
        logger.info(
            f"  - Current allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
        )
        logger.info(
            f"  - Maximum allocated: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB"
        )
        logger.info(
            f"  - Current reserved: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB"
        )
        logger.info(
            f"  - Maximum reserved: {torch.cuda.max_memory_reserved(device) / 1e6:.2f} MB"
        )
        # Reset peak stats for next run
        torch.cuda.reset_peak_memory_stats()

    return {
        "model": model,
        "final_stats": training_stats[-1] if training_stats else None,
        "training_history": training_stats,
        "test_metrics": test_metrics,
        "hardware_info": get_hardware_info(),
        "run_dir": run_dir,  # Include the run directory in the returned results
        "run_name": run_name,  # Include the run name for reference
    }
