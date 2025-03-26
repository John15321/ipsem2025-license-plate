"""Command-line interface for quantum-classical hybrid model."""

from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console

from ..utils.logging_utils import configure_logging, get_logger
from .test import evaluate_model
from .train import train_hybrid_model
from .utils import load_model

app = typer.Typer(
    help="IPSEM 2025 Hybrid Quantum-Classical Neural Network", add_completion=False
)
console = Console()
logger = get_logger(__name__)


@app.command("train")
def train_command(
    n_qubits: int = typer.Option(2, "--n-qubits", "-q", help="Number of qubits to use"),
    ansatz_reps: int = typer.Option(
        1, "--ansatz-reps", "-r", help="Depth of RealAmplitudes ansatz"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Training batch size"
    ),
    train_ratio: float = typer.Option(
        0.7, "--train-ratio", help="Ratio of data to use for training (0.0-1.0)"
    ),
    val_ratio: float = typer.Option(
        0.15, "--val-ratio", help="Ratio of data to use for validation (0.0-1.0)"
    ),
    learning_rate: float = typer.Option(
        1e-3, "--learning-rate", "-l", help="Learning rate"
    ),
    dataset_type: str = typer.Option(
        "emnist", "--dataset-type", "-d", help="Dataset type (emnist, mnist, or custom)"
    ),
    dataset_path: str = typer.Option(
        "data", "--dataset-path", "-p", help="Path to dataset"
    ),
    model_save_path: Optional[str] = typer.Option(
        None, "--model-save-path", "-m", help="Path to save trained model"
    ),
    stats_file: str = typer.Option(
        "training_stats.csv",
        "--stats-file",
        "-s",
        help="Path to save training statistics CSV",
    ),
    run_test: bool = typer.Option(
        False, "--test", "-t", help="Run evaluation on test set after training"
    ),
    log_file: str = typer.Option(None, "--log-file", help="Path to save log output"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> int:
    """Train a hybrid quantum-classical neural network."""
    try:
        # Configure logging based on verbosity
        log_level = "DEBUG" if verbose else "INFO"
        configure_logging(
            level=log_level,
            log_to_console=True,
            log_to_file=bool(log_file),
            log_file=log_file,
        )

        # Validate dataset split ratios
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError(
                f"Train ratio must be between 0.0 and 1.0, got {train_ratio}"
            )
        if not 0.0 <= val_ratio <= 1.0:
            raise ValueError(
                f"Validation ratio must be between 0.0 and 1.0, got {val_ratio}"
            )
        if train_ratio + val_ratio > 1.0:
            raise ValueError(
                f"Sum of train ratio ({train_ratio}) and validation ratio ({val_ratio}) cannot exceed 1.0"
            )

        logger.info("Starting quantum-classical model training...")
        logger.debug("Training parameters:")
        logger.debug("  n_qubits: %s", n_qubits)
        logger.debug("  ansatz_reps: %s", ansatz_reps)
        logger.debug("  epochs: %s", epochs)
        logger.debug("  batch_size: %s", batch_size)
        logger.debug("  train_ratio: %s", train_ratio)
        logger.debug("  val_ratio: %s", val_ratio)
        logger.debug("  dataset_type: %s", dataset_type)
        logger.debug("  model_save_path: %s", model_save_path)
        logger.debug("  stats_file: %s", stats_file)
        logger.debug("  run_test: %s", run_test)

        result = train_hybrid_model(
            n_qubits=n_qubits,
            ansatz_reps=ansatz_reps,
            epochs=epochs,
            batch_size=batch_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            learning_rate=learning_rate,
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            model_save_path=model_save_path,
            stats_file=stats_file,
            log_file=log_file,
            run_test=run_test,
            verbose=verbose,
        )
        logger.info("Training completed successfully")
        if result["test_metrics"]:
            logger.info(
                "Test accuracy: %.2f%%", result["test_metrics"]["test_accuracy"]
            )
        return 0
    except Exception as e:
        logger.exception(f"Training failed: %s", e)
        return 1


@app.command("test")
def test_command(
    model_path: str = typer.Option(
        ..., "--model-path", "-m", help="Path to saved model"
    ),
    dataset_type: str = typer.Option(
        "emnist", "--dataset-type", "-d", help="Dataset type (emnist, mnist, or custom)"
    ),
    dataset_path: str = typer.Option(
        "data", "--dataset-path", "-p", help="Path to dataset"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for testing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> int:
    """Evaluate a trained model on a test dataset."""
    try:
        # Configure logging
        log_level = "DEBUG" if verbose else "INFO"
        configure_logging(level=log_level)

        # Load model
        model_path = Path(model_path)
        model, metadata = load_model(model_path)
        logger.info("Loaded model from %s", model_path)
        logger.info("Model metadata: %s", metadata)

        # Load dataset
        from ..datasets.custom import CustomImageDataset
        from ..datasets.emnist import EMNISTDataset
        from ..datasets.mnist import MNISTDataset

        logger.info("Loading %s dataset from %s", dataset_type, dataset_path)
        if dataset_type.lower() == "emnist":
            dataset = EMNISTDataset(root=dataset_path, train=False, download=True)
        elif dataset_type.lower() == "mnist":
            dataset = MNISTDataset(root=dataset_path, train=False, download=True)
        elif dataset_type.lower() == "custom":
            dataset = CustomImageDataset(root=dataset_path)
        else:
            raise ValueError("Unknown dataset type: %s", dataset_type)

        # Create test loader
        _, _, test_loader = dataset.create_data_loaders(
            batch_size=batch_size,
            train_ratio=0.0,  # Use all data for testing
            val_ratio=0.0,
        )

        # Evaluate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        metrics = evaluate_model(model, test_loader, device)

        # Log results
        logger.info("Test Results:")
        for name, value in metrics.items():
            logger.info("%s: %.2f", name, value)

        return 0

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1


def main():
    """Main entry point for the quantum neural network CLI."""
    try:
        return app()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        return 1
