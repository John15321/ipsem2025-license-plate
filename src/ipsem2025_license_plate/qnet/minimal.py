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
from typing import Optional, Dict, Any, Union
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import typer
from ..utils.logging_utils import configure_logging, get_logger

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

        # Classical CNN feature extractor
        self.classical_net = nn.Sequential(
            # First Conv block
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second Conv block
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third Conv block
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and Dense layers
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),  # Input size computed for 28x28 MNIST images
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
    peak_memory = 0  # Initialize peak memory tracker
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            if batch_idx % 10 == 0:
                logger.debug(f"Processing batch {batch_idx}/{len(train_loader)}")
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Memory tracking
            if device == "cuda":
                current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                peak_memory = max(peak_memory, current_memory)
            
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                batch_time = time.time() - batch_start_time
                logger.debug(f"Batch {batch_idx} took {batch_time:.2f}s")
        
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


def train_hybrid_model(
    n_qubits: int = 2,
    ansatz_reps: int = 1,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    stats_file: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
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
        stats_file: Path to save training statistics CSV
        log_file: Path to save log output
        verbose: Enable verbose output
        
    Returns:
        Dict containing:
        - model: Trained HybridModel instance
        - final_stats: Statistics from final epoch
        - training_history: List of stats from all epochs
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
    
    # Setup data
    num_classes = calc_num_classes(n_qubits)
    digits = list(range(num_classes))
    
    train_data = get_mnist_subset(digits, train=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataset loaded: {len(train_data)} samples, {num_classes} classes")
    
    # Create model
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
    
    return {
        'model': model,
        'final_stats': training_stats[-1] if training_stats else None,
        'training_history': training_stats,
        'hardware_info': hw_info
    }

@app.command("train")
def train_command(
    n_qubits: int = typer.Option(2, "--n-qubits", "-q", help="Number of qubits to use"),
    ansatz_reps: int = typer.Option(1, "--ansatz-reps", "-r", help="Depth of RealAmplitudes ansatz"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", "-l", help="Learning rate"),
    stats_file: str = typer.Option(
        "training_stats.csv",
        "--stats-file",
        "-s",
        help="Path to save training statistics CSV"
    ),
    log_file: str = typer.Option(None, "--log-file", help="Path to save log output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> int:
    """Train a hybrid quantum-classical neural network on MNIST digits."""
    try:
        result = train_hybrid_model(
            n_qubits=n_qubits,
            ansatz_reps=ansatz_reps,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            stats_file=stats_file,
            log_file=log_file,
            verbose=verbose
        )
        logger.info("Training completed successfully")
        return 0
    except Exception as e:
        logger.exception("Training failed")
        return 1

def main():
    """Main entry point for the quantum neural network CLI."""
    try:
        return app()
    except Exception as e:
        logger.exception("An error occurred")
        return 1
