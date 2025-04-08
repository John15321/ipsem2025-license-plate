"""Hybrid quantum-classical model definition."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

from typing import Dict, Optional, Union

import qiskit_aer
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
# from qiskit.primitives import SamplerV2
from qiskit_aer.primitives import SamplerV2
from qiskit.primitives import BackendSamplerV2
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch import nn
from qiskit_aer import Aer

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

simulator_gpu = BackendSamplerV2(backend=Aer.get_backend(name='statevector_simulator', backend_options={"device": "GPU"}))




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
        n_qubits=6,  # Default to 6 qubits for better letter/number recognition
        ansatz_reps=2,  # Default to 2 repetitions for more expressive circuit
        num_classes=36,  # Default to 36 classes (10 digits + 26 letters)
        input_channels=1,
        sampler: Optional[Sampler] = None,
        use_gpu: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Set device - use provided device or auto-detect
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.info(
            "Initializing HybridModel with %s qubits, %s ansatz reps, %d classes on device: %s",
            n_qubits,
            ansatz_reps,
            num_classes,
            self.device,
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
        self.use_gpu = use_gpu and self.device.type == "cuda"

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
                128 * 8 * 8, 256
            ),  # Increased hidden layer size for 6-qubit support
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_qubits),
            # Activation ensures values are in proper range for quantum circuit
            nn.Tanh(),
        )

        # TODO

        # Create TorchConnector with quantum network
        logger.info("Creating TorchConnector for SamplerQNN")
        self.quantum_layer = TorchConnector(self.qnn)

        # Final classification layer - correctly sized for the QNN output
        logger.debug(
            "Creating final classifier layer: %s -> %s", qnn_output_dim, num_classes
        )

        # Classifier that handles the proper dimension
        self.classifier = nn.Linear(qnn_output_dim, num_classes)

        # Move model to specified device
        self.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
        logger.info("Model initialization complete")




    def create_quantum_network(self, n_qubits: int, ansatz_reps: int):
        """Creates the quantum circuit and sampler for the model."""
        # Quantum circuit setup
        logger.debug("Creating quantum feature map with %s qubits", n_qubits)
        # For 6 qubits, use a different repetition strategy to avoid overly complex circuits
        feature_map_reps = 1 if n_qubits > 4 else 2
        self.feature_map = ZZFeatureMap(
            feature_dimension=n_qubits, reps=feature_map_reps
        )

        logger.debug(
            "Creating RealAmplitudes ansatz with %s qubits, %s repetitions",
            n_qubits,
            ansatz_reps,
        )
        # For 6 qubits, use linear entanglement to reduce circuit depth
        entanglement = "linear" if n_qubits > 4 else "full"
        self.ansatz = RealAmplitudes(
            num_qubits=n_qubits, reps=ansatz_reps, entanglement=entanglement
        )

        # Combine feature map and ansatz
        circuit = QuantumCircuit(n_qubits)
        circuit.append(self.feature_map, range(n_qubits))
        circuit.append(self.ansatz, range(n_qubits))

        # Setup GPU-accelerated sampler if requested
        aer_simulator = None
        logger.info("use_gpu: %s", self.use_gpu)
        logger.info("Device: %s", self.device)
        if self.use_gpu:
            logger.info(
                "Using GPU-accelerated quantum simulator via qiskit-aer-gpu"
            )
            try:
                # Create simulator with GPU method
                aer_simulator = BackendSamplerV2(backend=Aer.get_backend(name='statevector_simulator', backend_options={"device": "GPU"}))

                # Create SamplerV2 with GPU backend
                self.sampler = SamplerV2(
                    backend_options={"method": "statevector"},
                    run_options={"device": "GPU"},
                )
                logger.info(
                    "GPU acceleration successfully enabled for quantum simulation"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize GPU quantum simulator: {e}")
                logger.info("Falling back to CPU-based quantum simulation")
                self.sampler = ()
                aer_simulator = None
        else:
            if use_gpu and self.device.type != "cuda":
                logger.warning(
                    "GPU requested but not available, falling back to CPU"
                )
            logger.info("Using CPU-based quantum simulator")
            self.sampler = SamplerV2()

        # Extract parameters from the circuit
        input_params = self.feature_map.parameters
        weight_params = self.ansatz.parameters

        # Set output shape - this is critical to fix the dimension mismatch
        # For 6 qubits, the state space is 2^6 = 64 dimensions
        qnn_output_dim = 2**n_qubits  # This will be 64 for 6 qubits
        logger.info(f"Using QNN output dimension of {qnn_output_dim}")

        # Quantum layer setup with correct output dimension
        logger.info(
            "Initializing SamplerQNN with input_gradients=True for proper backpropagation"
        )
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            sampler=self.sampler,
            input_gradients=True,  # Critical for hybrid model gradient flow
            sparse=False,  # Return dense probability array for proper backprop
            # No interpret function to ensure we get full 2^n_qubits output dimension
        )




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
               Expected dimensions: [batch_size, input_channels, 64, 64]

        Returns:
            Class probabilities of shape [batch_size, num_classes]
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

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
        logits = self.classifier(q_out_batch)

        # Apply log-softmax to ensure outputs are compatible with NLLLoss
        return torch.log_softmax(logits, dim=1)

    def get_circuit_depth(self) -> int:
        """Returns the depth of the quantum circuit used in the model."""
        return self.feature_map.depth() + self.ansatz.depth()

    def get_model_info(self) -> Dict[str, object]:
        """Returns a dictionary with information about the model configuration."""
        return {
            "n_qubits": self.n_qubits,
            "ansatz_reps": self.ansatz_reps,
            "num_classes": self.num_classes,
            "circuit_depth": self.get_circuit_depth(),
            "classical_params": sum(p.numel() for p in self.classical_net.parameters()),
            "quantum_params": len(self.ansatz.parameters),
            "total_params": sum(p.numel() for p in self.parameters()),
            "device": str(self.device),
            "using_gpu_quantum": self.use_gpu,
        }
