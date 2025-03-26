"""Hybrid quantum-classical model definition."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

from typing import Dict, Optional

import torch
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch import nn

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HybridModel(nn.Module):
    """Hybrid quantum-classical network for license plate image classification.

    Architecture:
    1. Classical CNN:
       - 3 Conv2D + MaxPool2D blocks (32, 64, 128 filters)
       - Dense layer with dropout
       - Output: n_qubits features
    2. Quantum Circuit: ZZFeatureMap + RealAmplitudes
    3. Classical Layer: Maps quantum output to class probabilities

    Input: 64x64 pixel images (after 3 MaxPool2D layers: 8x8)
    """

    def __init__(
        self,
        n_qubits=2,
        ansatz_reps=1,
        num_classes=2,
        input_channels=1,
        sampler: Optional[Sampler] = None,
    ):
        super().__init__()
        logger.info(
            "Initializing HybridModel with %s qubits, %s ansatz reps, and %d number of classes",
            n_qubits,
            ansatz_reps,
            num_classes,
        )

        # Input validation
        if n_qubits < 1:
            raise ValueError(f"Number of qubits must be positive, got {n_qubits}")
        if ansatz_reps < 1:
            raise ValueError(f"Ansatz repetitions must be positive, got {ansatz_reps}")
        if num_classes < 2:
            raise ValueError(f"Number of classes must be at least 2, got {num_classes}")
        if input_channels < 1:
            raise ValueError(f"Input channels must be positive, got {input_channels}")

        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.num_classes = num_classes

        # Classical CNN feature extractor for 64x64 pixel images
        # After 3 MaxPool2D layers (each dividing by 2): 64x64 -> 32x32 -> 16x16 -> 8x8
        # Final feature map size: 128 filters * 8 * 8 = 8192 features
        self.classical_net = nn.Sequential(
            # First Conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second Conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third Conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Flatten and Dense layers
            nn.Flatten(),
            nn.Linear(
                128 * 8 * 8, 128
            ),  # 8x8 is the size after 3 MaxPool2D layers from 64x64 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_qubits),
            # Activation ensures values are in proper range for quantum circuit
            nn.Tanh(),
        )

        # Quantum circuit setup
        logger.debug("Creating quantum feature map with %s qubits", n_qubits)
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)

        logger.debug(
            "Creating RealAmplitudes ansatz with %s qubits, %s repetitions",
            n_qubits,
            ansatz_reps,
        )
        self.ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps)

        # Combine feature map and ansatz
        circuit = self.feature_map.compose(self.ansatz)

        # Quantum layer setup
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            sampler=sampler or Sampler(),
            input_gradients=True,
        )
        self.quantum_layer = TorchConnector(self.qnn)

        # Final classification layer
        logger.debug(
            "Creating final classifier layer: %s -> %s", 2**n_qubits, num_classes
        )
        self.classifier = nn.Linear(2**n_qubits, num_classes)

        logger.info("Model initialization complete")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               Expected dimensions: [batch_size, input_channels, 64, 64]

        Returns:
            Class probabilities of shape [batch_size, num_classes]
        """
        # Classical CNN: image -> n_qubits features
        classical_out = self.classical_net(x)

        # Process batch through quantum layer (one sample at a time)
        batch_size = x.size(0)
        q_out_list = []

        for i in range(batch_size):
            q_out = self.quantum_layer(classical_out[i])
            q_out_list.append(q_out)

        # Combine quantum outputs back into a batch
        q_out_batch = torch.stack(q_out_list)

        # Classification: quantum output -> class probabilities
        return self.classifier(q_out_batch)

    def get_circuit_depth(self) -> int:
        """Returns the depth of the quantum circuit used in the model."""
        return self.feature_map.depth() + self.ansatz.depth()

    def get_model_info(self) -> Dict[str, any]:
        """Returns a dictionary with information about the model configuration."""
        return {
            "n_qubits": self.n_qubits,
            "ansatz_reps": self.ansatz_reps,
            "num_classes": self.num_classes,
            "circuit_depth": self.get_circuit_depth(),
            "classical_params": sum(p.numel() for p in self.classical_net.parameters()),
            "quantum_params": len(self.ansatz.parameters),
            "total_params": sum(p.numel() for p in self.parameters()),
        }
