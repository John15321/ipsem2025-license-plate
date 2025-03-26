"""Hybrid quantum-classical model definition."""

import torch.nn as nn
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


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
        logger.info(
            "Initializing HybridModel with %s qubits, %s ansatz reps, and %d number of classes",
            n_qubits,
            ansatz_reps,
            num_classes
        )

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
            nn.Linear(128, n_qubits),
        )

        # Quantum circuit setup
        logger.debug("Creating quantum feature map with %s qubits", n_qubits)
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits)

        logger.debug(
            "Creating RealAmplitudes ansatz with %s qubits, %s repetitions",
            n_qubits,
            ansatz_reps
        )
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
        logger.debug("Creating final classifier layer: %s -> %s", 2**n_qubits, num_classes)
        self.classifier = nn.Linear(2**n_qubits, num_classes)

        logger.info("Model initialization complete")

    def forward(self, x):
        # Classical CNN: image -> n_qubits features
        classical_out = self.classical_net(x)
        # Quantum processing: features -> 2^n_qubits vector
        q_out = self.quantum_layer(classical_out)
        # Classification: quantum output -> class probabilities
        return self.classifier(q_out)
