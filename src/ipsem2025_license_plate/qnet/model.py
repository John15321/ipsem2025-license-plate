"""Hybrid quantum-classical model definition."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import Aer
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch import nn

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# Define a function to create QNN using SamplerQNN
def create_qnn(n_qubits=6, ansatz_reps=2, feature_map_reps=1, use_gpu=True):
    """Creates a quantum neural network using the approach from the POC.

    Args:
        n_qubits: Number of qubits to use
        ansatz_reps: Number of repetitions in the ansatz
        feature_map_reps: Number of repetitions in the feature map
        use_gpu: Whether to use GPU acceleration for quantum simulation

    Returns:
        A SamplerQNN instance
    """
    logger.info(
        f"Creating quantum neural network with {n_qubits} qubits, {feature_map_reps} feature map reps, and {ansatz_reps} ansatz repetitions"
    )
    logger.info(f"Using SamplerQNN with GPU acceleration: {use_gpu}")

    # Create feature map and ansatz
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=feature_map_reps)
    entanglement = "linear" if n_qubits > 4 else "full"
    ansatz = RealAmplitudes(
        num_qubits=n_qubits, reps=ansatz_reps, entanglement=entanglement
    )

    # Combine into quantum circuit
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Create backend based on availability of GPU
    if use_gpu:
        try:
            logger.info("Attempting to use GPU-accelerated quantum simulator")
            # Create SamplerQNN implementation
            backend = Aer.get_backend(
                name="statevector_simulator", backend_options={"device": "GPU"}
            )
            simulator_gpu = BackendSamplerV2(backend=backend)

            # Transpile the circuit for the backend - important for GPU performance
            qc = transpile(qc, backend)

            # Verify if GPU is being used by Aer
            backend_config = backend.configuration()
            if hasattr(backend_config, "gpu") and backend_config.gpu:
                logger.info("Confirmed: Aer backend is configured to use GPU")
                logger.info(f"GPU configuration: {backend.configuration()}")
            else:
                logger.warning("Aer backend may not be using GPU despite configuration")

            qnn = SamplerQNN(
                circuit=qc,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                input_gradients=True,
                sampler=simulator_gpu,
                sparse=False,
            )
            logger.info("Successfully created GPU-accelerated SamplerQNN")
        except Exception as e:
            logger.warning(f"Could not create GPU QNN: {e}")
            logger.info("Falling back to CPU implementation")
            qnn = SamplerQNN(
                circuit=qc,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                input_gradients=True,
                sparse=False,
            )
    else:
        logger.info("Using CPU-based quantum simulator")
        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
            sparse=False,
        )

    logger.info(
        f"QNN created with {len(feature_map.parameters)} input params and {len(ansatz.parameters)} weight params"
    )
    return qnn


class HybridModel(nn.Module):
    """Hybrid quantum-classical network for license plate image classification.

    Architecture exactly matching the POC:
    1. Classical CNN layers (Conv1, Pool, Conv2, Pool, etc.)
    2. Dense layers (fc1, fc2)
    3. Quantum layer (QNN)
    4. Final classification layer (fc3)
    5. Additional processing (cat, softmax)
    """

    def __init__(
        self,
        n_qubits=6,  # Default to 6 qubits for better letter/number recognition
        ansatz_reps=2,  # Default to 2 repetitions for more expressive circuit
        feature_map_reps=1,  # Default to 1 repetition for feature map
        num_classes=36,  # Default to 36 classes (10 digits + 26 letters)
        input_channels=1,
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
            "Initializing HybridModel with %s qubits, %s feature map reps, %s ansatz reps, %d classes on device: %s",
            n_qubits,
            feature_map_reps,
            ansatz_reps,
            num_classes,
            self.device,
        )

        # Log detailed GPU information if using CUDA
        if self.device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(self.device)}")
            logger.info(
                f"CUDA Memory: Total {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB"
            )
            logger.info(
                f"CUDA Capability: {torch.cuda.get_device_capability(self.device)}"
            )
            logger.info(
                f"Current memory allocated: {torch.cuda.memory_allocated(self.device) / 1e6:.2f} MB"
            )
            logger.info(
                f"Current memory reserved: {torch.cuda.memory_reserved(self.device) / 1e6:.2f} MB"
            )

        # Input validation
        if n_qubits < 1:
            raise ValueError(f"Number of qubits must be positive, got {n_qubits}")
        if ansatz_reps < 1:
            raise ValueError(f"Ansatz repetitions must be positive, got {ansatz_reps}")
        if feature_map_reps < 1:
            raise ValueError(
                f"Feature map repetitions must be positive, got {feature_map_reps}"
            )
        if num_classes < 2:
            raise ValueError(f"Number of classes must be at least 2, got {num_classes}")
        if input_channels < 1:
            raise ValueError(f"Input channels must be positive, got {input_channels}")

        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.feature_map_reps = feature_map_reps
        self.num_classes = num_classes
        self.use_gpu = use_gpu and self.device.type == "cuda"

        # Standard configuration for 64x64 EMNIST images with batch size 64
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5).to(self.device)
        logger.info(f"conv1 created on device: {self.conv1.weight.device}")
        # After first convolution: (64-5+0)/1 + 1 = 60x60x6
        # After first pooling: 30x30x6

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5).to(self.device)
        logger.info(f"conv2 created on device: {self.conv2.weight.device}")
        # After second convolution: (30-5+0)/1 + 1 = 26x26x16
        # After second pooling: 13x13x16 = 2704 features

        self.dropout = nn.Dropout2d().to(self.device)

        # Fixed fully connected layer sizes for 64x64 input images
        # Corrected: Use actual flattened size 13x13x16 = 2704 as input dimension for fc1
        self.fc1 = nn.Linear(2704, 64).to(self.device)  # 13x13x16 = 2704 input features
        logger.info(f"fc1 created on device: {self.fc1.weight.device}")

        self.fc2 = nn.Linear(64, n_qubits).to(self.device)
        logger.info(f"fc2 created on device: {self.fc2.weight.device}")

        # Create quantum neural network using the POC approach
        logger.info(f"Creating quantum neural network...")
        qnn = create_qnn(
            n_qubits=n_qubits,
            ansatz_reps=ansatz_reps,
            feature_map_reps=feature_map_reps,
            use_gpu=self.use_gpu,
        )

        # Create TorchConnector with quantum network
        logger.info("Creating TorchConnector for SamplerQNN")
        self.qnn = TorchConnector(qnn)

        # Calculate the output dimension of the quantum network for final layer sizing
        # SamplerQNN returns 2^n_qubits values
        qnn_output_dim = 2**n_qubits

        logger.info(f"QNN output dimension: {qnn_output_dim}")

        # Remaining classical layers
        self.fc3 = nn.Linear(qnn_output_dim, num_classes).to(self.device)
        logger.info(f"fc3 created on device: {self.fc3.weight.device}")

        # Explicitly move all modules to device
        self.to(self.device)

        # Log all parameters and their devices for verification
        logger.info("Parameter device placement:")
        for name, param in self.named_parameters():
            logger.info(f"Parameter '{name}' is on device: {param.device}")

        logger.info(f"Model moved to device: {self.device}")
        logger.info("Model initialization complete")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Class probabilities of shape [batch_size, num_classes]
        """
        # Ensure input is on the correct device
        x = x.to(self.device, non_blocking=True)
        batch_size = x.shape[0]

        # Debug input shape
        # logger.debug(f"Input shape: {x.shape}")

        # Log the input shape to verify dimensions
        # logger.debug(f"Input batch shape: {x.shape}")

        # Forward pass through conv layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # logger.debug(f"After conv1+pool: {x.shape}")

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # logger.debug(f"After conv2+pool: {x.shape}")

        x = self.dropout(x)

        # Flatten layer
        x = x.view(batch_size, -1)
        # logger.debug(f"After flatten: {x.shape}")

        # Dense layers
        x = F.relu(self.fc1(x))
        # logger.debug(f"After fc1: {x.shape}")

        x = self.fc2(x)
        # logger.debug(f"After fc2: {x.shape}")

        # Apply QNN
        x = self.qnn(x)
        # logger.debug(f"After QNN: {x.shape}")

        # Final classical layer
        x = self.fc3(x)
        # logger.debug(f"After fc3: {x.shape}")

        # Return logits directly instead of applying log_softmax
        return x

    def get_circuit_depth(self) -> int:
        """Returns the depth of the quantum circuit used in the model."""
        # Fix the access to the quantum circuit
        # The TorchConnector doesn't have a 'qnn' attribute, it wraps the QNN directly
        # We need to use the correct attribute to access the underlying quantum neural network
        try:
            if hasattr(self.qnn, "quantum_neural_network"):
                # For newer versions of TorchConnector
                qnn = self.qnn.quantum_neural_network
                if hasattr(qnn, "circuit"):
                    return qnn.circuit.depth()
            elif hasattr(self.qnn, "_model"):
                # For older versions of TorchConnector
                qnn = self.qnn._model
                if hasattr(qnn, "circuit"):
                    return qnn.circuit.depth()

            # If we can't access the circuit directly, log this fact
            logger.warning("Could not access circuit depth from TorchConnector")
            return 0  # Return a default value
        except Exception as e:
            logger.error(f"Error accessing circuit depth: {e}")
            return 0  # Return a default value in case of any error

    def get_model_info(self) -> Dict[str, object]:
        """Returns a dictionary with information about the model configuration."""

        # Get quantum parameters count safely
        quantum_params = 0
        try:
            # Try different ways to access the underlying QNN parameters
            if hasattr(self.qnn, "quantum_neural_network"):
                qnn = self.qnn.quantum_neural_network
                if hasattr(qnn, "weight_params"):
                    quantum_params = len(qnn.weight_params)
            elif hasattr(self.qnn, "_model"):
                qnn = self.qnn._model
                if hasattr(qnn, "weight_params"):
                    quantum_params = len(qnn.weight_params)
            else:
                # If we can't determine exact count, estimate from the model parameter count
                quantum_params = sum(
                    p.numel()
                    for name, p in self.named_parameters()
                    if name.startswith("qnn")
                )
        except Exception as e:
            logger.error(f"Error determining quantum parameter count: {e}")

        # Calculate classical parameters by subtracting quantum ones from total
        total_params = sum(p.numel() for p in self.parameters())
        classical_params = total_params - quantum_params

        return {
            "n_qubits": self.n_qubits,
            "ansatz_reps": self.ansatz_reps,
            "feature_map_reps": self.feature_map_reps,
            "num_classes": self.num_classes,
            "circuit_depth": self.get_circuit_depth(),
            "classical_params": classical_params,
            "quantum_params": quantum_params,
            "total_params": total_params,
            "device": str(self.device),
            "using_gpu_quantum": self.use_gpu,
            "layer_devices": {
                name: str(param.device) for name, param in self.named_parameters()
            },
        }
