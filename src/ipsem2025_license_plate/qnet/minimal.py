#!/usr/bin/env python3
"""
Hybrid Quantum-Classical MNIST classifier using PyTorch and Qiskit.

The network combines a classical CNN front-end with a quantum circuit middle layer,
followed by a classical output layer. The quantum circuit uses n_qubits to process
features and outputs a 2^n_qubits dimensional vector.

Usage:
    ipsem2025-qnet train [--n-qubits N] [--ansatz-reps R] [--epochs E] [--batch-size B]
"""

import sys
from pathlib import Path

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

# CLI setup
app = typer.Typer(
    help="Hybrid Quantum-Classical Neural Network for MNIST",
    add_completion=False,
)
logger = get_logger(__name__)


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
    1. Classical CNN: Compresses MNIST image to n_qubits features
    2. Quantum Circuit: Processes features using parameterized gates
    3. Classical Layer: Maps quantum output to class probabilities
    """
    def __init__(self, n_qubits=2, ansatz_reps=1, num_classes=2):
        super().__init__()
        logger.info(f"Initializing HybridModel with {n_qubits} qubits, {ansatz_reps} ansatz reps")

        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.num_classes = num_classes

        # Classical feature extractor
        self.classical_net = nn.Sequential(
            nn.Flatten(),            
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)  
        )

        # Quantum circuit setup
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits)
        self.ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps)
        circuit = self.feature_map.compose(self.ansatz)

        # Quantum layer that outputs 2^n_qubits-dim vector
        sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            sampler=sampler,
        )
        self.quantum_layer = TorchConnector(self.qnn)

        # Final classification layer
        self.classifier = nn.Linear(2 ** n_qubits, num_classes)
        
        logger.info("Model initialization complete")

    def forward(self, x):
        # Classical preprocessing: image -> n_qubits features
        classical_out = self.classical_net(x)
        # Quantum processing: features -> 2^n_qubits vector
        q_out = self.quantum_layer(classical_out)
        # Classification: quantum output -> class probabilities
        return self.classifier(q_out)


def train_model(model, train_loader, device="cpu", epochs=3):
    """Train the hybrid model using cross entropy loss and Adam optimizer."""
    logger.info(f"Starting training for {epochs} epochs on {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                logger.debug(f"Processing batch {batch_idx}/{len(train_loader)}")
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    logger.info("Training complete")


# CLI implementation
@app.command("train")
def train_command(
    n_qubits: int = typer.Option(
        2,
        "--n-qubits",
        "-q",
        help="Number of qubits to use"
    ),
    ansatz_reps: int = typer.Option(
        1,
        "--ansatz-reps",
        "-r",
        help="Depth (reps) of RealAmplitudes ansatz"
    ),
    epochs: int = typer.Option(
        3,
        "--epochs",
        "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Training batch size"
    ),
    log_file: str = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to save log output (defaults to console only)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
):
    """
    Train a hybrid quantum-classical neural network on MNIST digits.
    
    The number of digits used is determined by the number of qubits (min(2^n_qubits, 10)).
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
    logger.info(f"Configuration: n_qubits={n_qubits}, ansatz_reps={ansatz_reps}, epochs={epochs}, batch_size={batch_size}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Determine how many classes based on qubits
    num_classes = calc_num_classes(n_qubits)
    digits = list(range(num_classes))  # e.g. [0,1] or [0,1,2,3], etc.
    logger.info(f"Using {num_classes} classes (digits {digits})")

    # Load subset of MNIST with these digits
    logger.info("Loading MNIST dataset")
    train_data = get_mnist_subset(digits, train=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataset loaded with {len(train_data)} samples")

    # Build model
    logger.info("Building hybrid quantum-classical model")
    model = HybridModel(
        n_qubits=n_qubits, 
        ansatz_reps=ansatz_reps,
        num_classes=num_classes
    )
    
    # Log model structure details
    logger.debug(f"Model structure: Classical preprocessing (MNIST → {n_qubits} features) → " +
                f"Quantum circuit ({n_qubits} qubits, {ansatz_reps} ansatz reps) → " +
                f"Classical output layer ({2**n_qubits} → {num_classes})")

    # Train model
    train_model(model, train_loader, device=device, epochs=epochs)
    
    logger.info("Training completed successfully")
    return 0


def main():
    """Main entry point for the quantum neural network CLI."""
    try:
        return app()
    except Exception as e:
        logger.exception("An error occurred during execution")
        return 1

