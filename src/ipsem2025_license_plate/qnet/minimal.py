#!/usr/bin/env python3
"""
Hybrid Quantum-Classical MNIST classifier using PyTorch and Qiskit.

The network combines a classical CNN front-end with a quantum circuit middle layer,
followed by a classical output layer. The quantum circuit uses n_qubits to process 
features and outputs a 2^n_qubits dimensional vector.
"""

import sys
from pathlib import Path
import csv
from datetime import datetime
import time
import platform
import psutil
import fcntl
from typing import Optional, Dict, Any, Union, Tuple
import os
import json
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms

import typer
from ..utils.logging_utils import configure_logging, get_logger
from ..datasets.mnist import MNISTDataset
from ..datasets.emnist import EMNISTDataset
from ..datasets.custom import CustomImageDataset
from ..datasets.base import BaseDataset

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

app = typer.Typer(
    help="Hybrid Quantum-Classical Neural Network for MNIST",
    add_completion=False,
)
logger = get_logger(__name__)

def get_hardware_info() -> Dict[str, str]:
    """Collect system hardware information."""
    return {
        'cpu_model': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'total_memory': f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
        'cpu_count': str(psutil.cpu_count(logical=False)),
        'cpu_threads': str(psutil.cpu_count(logical=True))
    }

def log_training_stats(stats_file: Path, stats: dict):
    """Log training statistics to CSV file with file locking for safety."""
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(stats_file, mode='a', newline='') as f:
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
        logger.error(f"Failed to write training stats: {e}")

def calc_num_classes(n_qubits: int) -> int:
    """Calculate number of output classes based on qubit count, capped at 10."""
    return min(2 ** n_qubits, 10)


def get_mnist_subset(digits, train=True):
    """Load a subset of MNIST containing only specified digits.
    
    Args:
        digits: List of digit labels to include
        train: Whether to load training or test set
        
    Returns:
        Dataset with remapped labels from 0 to len(digits)-1
    """
    logger.info(f"Loading MNIST subset with digits: {digits}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    digit_to_new_label = {digit: idx for idx, digit in enumerate(digits)}
    
    # Filter dataset to keep only specified digits
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in digits:
            indices.append(i)
    
    logger.info(f"Found {len(indices)} samples for digits {digits}")
    subset_data = Subset(dataset, indices)

    # Wrapper class to remap original labels to sequential indices
    class RemappedSubset(Subset):
        def __init__(self, subset, label_map):
            super().__init__(subset.dataset, subset.indices)
            self.label_map = label_map

        def __getitem__(self, idx):
            x, orig_label = super().__getitem__(idx)
            return x, self.label_map[int(orig_label)]

    return RemappedSubset(subset_data, digit_to_new_label)


class HybridModel(nn.Module):
    """Hybrid quantum-classical network for MNIST digit classification.
    
    Architecture:
    1. Classical CNN: 
       - 3 Conv2D + MaxPool2D blocks
       - Dense layer with dropout
       - Output: n_qubits features
    2. Quantum Circuit: ZZFeatureMap + RealAmplitudes
    3. Classical Layer: Maps quantum output to class probabilities
    """
    def __init__(self, n_qubits=2, ansatz_reps=1, num_classes=2):
        super().__init__()
        logger.info(f"Initializing HybridModel with {n_qubits} qubits, {ansatz_reps} ansatz reps")

        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.num_classes = num_classes
        
        # Calculate dimensions for 64x64 input
        # After 3 MaxPool2D layers (each dividing by 2): 64 -> 32 -> 16 -> 8
        # After 3 Conv2D layers (each reducing by 2): 8 -> 6 -> 4 -> 2
        # Final feature map size will be 128 * 2 * 2 = 512

        # Classical CNN feature extractor
        self.classical_net = nn.Sequential(
            # First Conv block: 64x64 -> 31x31 -> 15x15
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second Conv block: 15x15 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third Conv block: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and Dense layers
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),  # 8x8 is the size after 3 MaxPool2D layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_qubits)
        )

        # Quantum circuit setup
        logger.debug(f"Creating quantum feature map with {n_qubits} qubits")
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits)
        
        logger.debug(f"Creating RealAmplitudes ansatz with {n_qubits} qubits, {ansatz_reps} repetitions")
        self.ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps)
        circuit = self.feature_map.compose(self.ansatz)

        # Quantum layer setup
        logger.debug("Creating SamplerQNN")
        sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            sampler=sampler,
        )
        self.quantum_layer = TorchConnector(self.qnn)

        # Final classification layer
        logger.debug(f"Creating final classifier layer: {2**n_qubits} -> {num_classes}")
        self.classifier = nn.Linear(2 ** n_qubits, num_classes)
        
        logger.info("Model initialization complete")

    def forward(self, x):
        # Classical CNN: image -> n_qubits features
        classical_out = self.classical_net(x)
        # Quantum processing: features -> 2^n_qubits vector
        q_out = self.quantum_layer(classical_out)
        # Classification: quantum output -> class probabilities
        return self.classifier(q_out)


def train_model(model, train_loader, device="cpu", epochs=3, stats_file: Optional[Path] = None):
    """Train the hybrid model using cross entropy loss and Adam optimizer."""
    logger.info(f"Starting training for {epochs} epochs on {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr = 1e-3
    
    model.to(device)
    model.train()
    
    # Track hardware info
    hardware_info = get_hardware_info()
    logger.info(f"Hardware info: {hardware_info}")
    
    training_stats = []
    total_start_time = time.time()
    peak_memory = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
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
            
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        
        # Collect and immediately save epoch statistics
        epoch_stats = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'epoch_time': epoch_time,
            'total_time': time.time() - total_start_time,
            'learning_rate': lr,
            'cpu_memory_mb': psutil.Process().memory_info().rss / 1024**2,
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0,
            'batch_size': train_loader.batch_size,
            **hardware_info
        }
        
        training_stats.append(epoch_stats)
        
        # Immediately write epoch stats if stats_file is provided
        if stats_file is not None:
            log_training_stats(stats_file, epoch_stats)
        
        logger.info(
            f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, "
            f"Time={epoch_time:.2f}s"
        )
    
    logger.info(f"Training complete! Total time: {time.time() - total_start_time:.2f}s")
    return training_stats


def save_model(model: HybridModel, save_path: Path, metadata: dict):
    """Save model and its metadata to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state and metadata
    model_data = {
        'state_dict': model.state_dict(),
        'metadata': metadata
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Model saved to {save_path}")


def load_model(load_path: Path) -> Tuple[HybridModel, dict]:
    """Load model and its metadata from disk."""
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)
    
    metadata = model_data['metadata']
    
    model = HybridModel(
        n_qubits=metadata['n_qubits'],
        ansatz_reps=metadata['ansatz_reps'],
        num_classes=metadata['num_classes']
    )
    model.load_state_dict(model_data['state_dict'])
    
    logger.info(f"Model loaded from {load_path}")
    return model, metadata


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluate model on test dataset."""
    logger.info("Starting model evaluation")
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    class_correct = torch.zeros(model.num_classes)
    class_total = torch.zeros(model.num_classes)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Calculate metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(model.num_classes):
        if class_total[i] > 0:
            class_accuracies[f"class_{i}_accuracy"] = 100.0 * class_correct[i].item() / class_total[i].item()
    
    metrics = {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        **class_accuracies
    }
    
    logger.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return metrics


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
    stats_file: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
    run_test: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train the hybrid quantum-classical model on MNIST digits.
    
    Args:
        n_qubits: Number of qubits in quantum circuit
        ansatz_reps: Number of repetitions in quantum ansatz
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for Adam optimizer
        device: Device to train on ('cuda' or 'cpu'), defaults to auto-detect
        dataset_type: Type of dataset to use ('emnist', 'mnist', or 'custom')
        dataset_path: Path to dataset
        model_save_path: Path to save trained model
        stats_file: Path to save training statistics CSV
        log_file: Path to save log output
        run_test: Whether to run evaluation on test set after training
        verbose: Enable verbose output
        
    Returns:
        Dict containing:
        - model: Trained HybridModel instance
        - final_stats: Statistics from final epoch
        - training_history: List of stats from all epochs
        - test_metrics: Evaluation metrics on test set
        - hardware_info: System information
    """
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(
        level=log_level,
        log_to_console=True,
        log_to_file=log_file is not None,
        log_file=log_file
    )
    
    logger.info("Starting hybrid quantum-classical MNIST training")
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Collect hardware info
    hw_info = get_hardware_info()
    logger.info(f"Hardware info: {hw_info}")
    
    # Load appropriate dataset
    logger.info(f"Loading {dataset_type} dataset from {dataset_path}")
    if dataset_type.lower() == "emnist":
        dataset = EMNISTDataset(root=dataset_path, train=True, download=True)
    elif dataset_type.lower() == "mnist":
        dataset = MNISTDataset(root=dataset_path, train=True, download=True)
    elif dataset_type.lower() == "custom":
        dataset = CustomImageDataset(root=dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create data loaders with train/val/test split
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=4
    )
    logger.info(f"Created data loaders - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)} samples")
    
    # Create model with correct number of classes for dataset
    num_classes = dataset.get_num_classes()
    model = HybridModel(
        n_qubits=n_qubits,
        ansatz_reps=ansatz_reps,
        num_classes=num_classes
    )
    
    # Convert stats_file to Path if provided
    stats_path = Path(stats_file) if stats_file else None
    
    # Train model with stats_file parameter
    training_stats = train_model(
        model, 
        train_loader, 
        device=device, 
        epochs=epochs,
        stats_file=stats_path
    )
    
    # Run test if requested
    test_metrics = None
    if run_test:
        logger.info("Running model evaluation on test set...")
        test_metrics = evaluate_model(model, test_loader, device)
    
    # Save model if path provided
    if model_save_path:
        save_path = Path(model_save_path)
        metadata = {
            'n_qubits': n_qubits,
            'ansatz_reps': ansatz_reps,
            'num_classes': num_classes,
            'dataset_type': dataset_type,
            'test_accuracy': test_metrics['test_accuracy'] if test_metrics else None,
            'training_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': device,
            'timestamp': datetime.now().isoformat()
        }
        save_model(model, save_path, metadata)
    
    return {
        'model': model,
        'final_stats': training_stats[-1] if training_stats else None,
        'training_history': training_stats,
        'test_metrics': test_metrics,
        'hardware_info': hw_info
    }

@app.command("train")
def train_command(
    n_qubits: int = typer.Option(2, "--n-qubits", "-q", help="Number of qubits to use"),
    ansatz_reps: int = typer.Option(1, "--ansatz-reps", "-r", help="Depth of RealAmplitudes ansatz"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", "-l", help="Learning rate"),
    dataset_type: str = typer.Option(
        "emnist",
        "--dataset-type", 
        "-d",
        help="Dataset type (emnist, mnist, or custom)"
    ),
    dataset_path: str = typer.Option(
        "data",
        "--dataset-path",
        "-p",
        help="Path to dataset"
    ),
    model_save_path: Optional[str] = typer.Option(
        None,
        "--model-save-path",
        "-m",
        help="Path to save trained model"
    ),
    stats_file: str = typer.Option(
        "training_stats.csv",
        "--stats-file",
        "-s",
        help="Path to save training statistics CSV"
    ),
    run_test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Run evaluation on test set after training"
    ),
    log_file: str = typer.Option(None, "--log-file", help="Path to save log output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> int:
    """Train a hybrid quantum-classical neural network."""
    try:
        result = train_hybrid_model(
            n_qubits=n_qubits,
            ansatz_reps=ansatz_reps,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            model_save_path=model_save_path,
            stats_file=stats_file,
            log_file=log_file,
            run_test=run_test,
            verbose=verbose
        )
        logger.info("Training completed successfully")
        if result['test_metrics']:
            logger.info(f"Test accuracy: {result['test_metrics']['test_accuracy']:.2f}%")
        return 0
    except Exception as e:
        logger.exception("Training failed")
        return 1


@app.command("test")
def test_command(
    model_path: str = typer.Option(
        ..., 
        "--model-path", 
        "-m", 
        help="Path to saved model"
    ),
    dataset_type: str = typer.Option(
        "emnist",
        "--dataset-type",
        "-d",
        help="Dataset type (emnist, mnist, or custom)"
    ),
    dataset_path: str = typer.Option(
        "data",
        "--dataset-path",
        "-p",
        help="Path to dataset"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for testing"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
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
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model metadata: {metadata}")
        
        # Load dataset
        logger.info(f"Loading {dataset_type} dataset from {dataset_path}")
        if dataset_type.lower() == "emnist":
            dataset = EMNISTDataset(root=dataset_path, train=False, download=True)
        elif dataset_type.lower() == "mnist":
            dataset = MNISTDataset(root=dataset_path, train=False, download=True)
        elif dataset_type.lower() == "custom":
            dataset = CustomImageDataset(root=dataset_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Create test loader
        _, _, test_loader = dataset.create_data_loaders(
            batch_size=batch_size,
            train_ratio=0.0,  # Use all data for testing
            val_ratio=0.0
        )
        
        # Evaluate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        metrics = evaluate_model(model, test_loader, device)
        
        # Log results
        logger.info("Test Results:")
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.2f}")
        
        return 0
        
    except Exception as e:
        logger.exception("Evaluation failed")
        return 1

def main():
    """Main entry point for the quantum neural network CLI."""
    try:
        return app()
    except Exception as e:
        logger.exception("An error occurred")
        return 1
