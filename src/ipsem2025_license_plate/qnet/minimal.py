#!/usr/bin/env python3
"""
Hybrid Quantum-Classical MNIST Example (Parameterized)

This script trains a small hybrid network on a subset of MNIST digits,
where the number of classes is min(2^n_qubits, 10). 

Usage:
    python hybrid_mnist.py [--n-qubits N] [--ansatz-reps R] [--epochs E] [--batch-size B]
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Qiskit / Qiskit ML imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


###############################################################################
# 1) Data Utilities
###############################################################################
def calc_num_classes(n_qubits: int) -> int:
    """
    Return how many classes (digits) we can handle, limited by MNIST's 10 classes.
    We pick min(2^n_qubits, 10).
    """
    return min(2 ** n_qubits, 10)


def get_mnist_subset(digits, train=True):
    """
    Return an MNIST subset containing only the given list of digit labels,
    and automatically remap labels to 0..(len(digits)-1).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)

    # Create a label->index mapping so that the labels become 0..(num_classes-1)
    digit_to_new_label = {digit: idx for idx, digit in enumerate(digits)}

    # Filter for only the specified digits
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in digits:
            indices.append(i)

    # Create the initial Subset
    subset_data = Subset(dataset, indices)

    # Subclass that remaps the label in __getitem__
    class RemappedSubset(Subset):
        def __init__(self, subset, label_map):
            # subset is already a Subset. We pass its .dataset and .indices to super().__init__()
            super().__init__(subset.dataset, subset.indices)
            self.label_map = label_map

        def __getitem__(self, idx):
            x, orig_label = super().__getitem__(idx)
            # Remap the original label to 0..(len(digits)-1)
            return x, self.label_map[int(orig_label)]

    # Return the newly constructed RemappedSubset
    return RemappedSubset(subset_data, digit_to_new_label)



###############################################################################
# 2) Define the Hybrid Model
###############################################################################
class HybridModel(nn.Module):
    """
    Hybrid model:
      - small classical MLP
      - quantum part (ZZFeatureMap + RealAmplitudes ansatz)
      - final classical layer for multi-class classification
    """
    def __init__(self, n_qubits=2, ansatz_reps=1, num_classes=2):
        """
        Args:
            n_qubits: Number of qubits in the quantum circuit.
            ansatz_reps: Repetitions (depth) in RealAmplitudes ansatz.
            num_classes: How many output classes for final classification.
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.num_classes = num_classes

        # ---- Classical feature extractor (small MLP) ----
        self.classical_net = nn.Sequential(
            nn.Flatten(),            # [batch, 1, 28, 28] -> [batch, 784]
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)  # output dimension = #qubits
        )

        # ---- Quantum circuit components ----
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits)
        self.ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps)

        # Compose them into one circuit
        circuit = self.feature_map.compose(self.ansatz)

        # SamplerQNN -> returns 2^n_qubits-dimensional probability distribution
        sampler = Sampler()  # or Sampler(options={"shots": 1024})
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            sampler=sampler,
        )

        # TorchConnector: turns QNN into a trainable PyTorch module
        self.quantum_layer = TorchConnector(self.qnn)

        # Final classical layer: from 2^n_qubits -> num_classes
        self.classifier = nn.Linear(2 ** n_qubits, num_classes)

    def forward(self, x):
        # 1) classical pre-processing -> [batch_size, n_qubits]
        classical_out = self.classical_net(x)
        # 2) quantum layer -> [batch_size, 2^n_qubits]
        q_out = self.quantum_layer(classical_out)
        # 3) final linear -> [batch_size, num_classes]
        logits = self.classifier(q_out)
        return logits


###############################################################################
# 3) Training Function
###############################################################################
def train_model(model, train_loader, device="cpu", epochs=3):
    """
    Standard PyTorch training loop for classification.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            logits = model(images)
            loss = criterion(logits, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete!")


###############################################################################
# 4) Main / Argument Parsing
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Hybrid QML on MNIST with variable qubits & ansatz depth")
    parser.add_argument("--n-qubits", type=int, default=2, help="Number of qubits to use")
    parser.add_argument("--ansatz-reps", type=int, default=1, help="Depth (reps) of RealAmplitudes ansatz")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Determine how many classes based on qubits
    num_classes = calc_num_classes(args.n_qubits)
    digits = list(range(num_classes))  # e.g. [0,1] or [0,1,2,3], etc.

    # Load subset of MNIST with these digits
    train_data = get_mnist_subset(digits, train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = HybridModel(
        n_qubits=args.n_qubits, 
        ansatz_reps=args.ansatz_reps,
        num_classes=num_classes
    )

    # Train model
    print(f"[INFO] Training on digits={digits} with {num_classes} classes")
    print(f"[INFO] n_qubits={args.n_qubits}, ansatz_reps={args.ansatz_reps}, epochs={args.epochs}")
    train_model(model, train_loader, device=device, epochs=args.epochs)


if __name__ == "__main__":
    main()
