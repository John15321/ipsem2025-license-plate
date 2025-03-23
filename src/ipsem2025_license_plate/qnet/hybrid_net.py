"""
Hybrid Quantum-Classical Neural Network Implementation.

This module implements a hybrid quantum-classical model for character recognition that combines:
1. A classical Convolutional Neural Network (CNN) for feature extraction
2. A quantum circuit for processing selected features
3. A final classical layer for classification

The architecture follows an approach where:
- CNN extracts visual features from the input images (64x64 grayscale)
- Selected features are encoded into quantum states using a feature map
- A parameterized quantum circuit processes the quantum features
- The quantum output is mapped back to classical probabilities for classification

Key Components:
- Classical CNN: 3 Conv2D layers with max pooling and dropout
- Quantum Feature Map: Configurable encoding of classical features into quantum states
- Quantum Ansatz: Trainable quantum circuit with rotation and entangling gates
- Output Layer: Maps quantum measurements to 36 classes (0-9, A-Z)

Benefits of this hybrid approach:
1. Classical CNN handles the initial visual feature extraction efficiently
2. Quantum circuit can capture complex feature interactions
3. Architecture is configurable for experimentation and optimization
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

# Qiskit-related imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2, n_local, pauli_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.visualization import circuit_drawer
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn, optim

from ..utils.logging_utils import get_logger
from .dataset_utils import load_and_prepare_dataset

logger = get_logger(__name__)


@dataclass
class QuantumArchitectureConfig:
    """Configuration for the quantum part of the hybrid network.

    This class defines all parameters that control the quantum circuit architecture:
    1. Feature Map Configuration:
       - Number of qubits for encoding classical features (default: 5)
       - Repetitions of the feature mapping circuit
       - Entanglement pattern between qubits

    2. Ansatz Configuration:
       - Number of repetitions of the variational circuit
       - Entanglement pattern for the ansatz

    3. Gate Count Targets (Optional):
       - Specific numbers of different gate types
       - Used to match paper specifications or experiment with architectures

    Example:
        >>> config = QuantumArchitectureConfig(
        ...     num_qubits=5,
        ...     feature_map_entanglement="linear",
        ...     ansatz_entanglement="circular",
        ...     feature_map_reps=2,
        ...     ansatz_reps=3
        ... )
    """

    # Number of qubits (determines feature dimension)
    num_qubits: int = 5  # Changed from 6 to 5 qubits

    # Feature map configuration
    feature_map_reps: int = 2  # Number of repetitions in feature map
    feature_map_entanglement: Literal["full", "linear", "circular"] = "full"
    feature_map_hadamard_layers: int = 2  # Number of Hadamard layers

    # Ansatz configuration
    ansatz_reps: int = 1  # Number of repetitions in RealAmplitudes
    ansatz_entanglement: Literal["full", "linear", "circular"] = "full"

    # Optional gate count targets adjusted for 5 qubits
    target_hadamard_gates: Optional[int] = 10
    target_phase_gates: Optional[int] = 35
    target_cnot_gates: Optional[int] = 50
    target_ry_gates: Optional[int] = 10
    target_ansatz_cnot_gates: Optional[int] = 4

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        if self.feature_map_reps < 1:
            raise ValueError("feature_map_reps must be positive")
        if self.ansatz_reps < 1:
            raise ValueError("ansatz_reps must be positive")

        logger.info("Created quantum architecture configuration:")
        logger.info("- Number of qubits: %d", self.num_qubits)
        logger.info(
            "- Feature map: %s entanglement, %d repetitions",
            self.feature_map_entanglement,
            self.feature_map_reps,
        )
        logger.info(
            "- Ansatz: %s entanglement, %d repetitions",
            self.ansatz_entanglement,
            self.ansatz_reps,
        )

    def save(self, path: str) -> None:
        """Save configuration to JSON file for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Saved quantum configuration to: %s", path)

    @classmethod
    def load(cls, path: str) -> "QuantumArchitectureConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        config = cls(**config_dict)
        logger.info("Loaded quantum configuration from: %s", path)
        return config

    def get_total_gates(self) -> Dict[str, int]:
        """Calculate total number of gates in the circuit.

        This method computes the minimum number of gates needed based on:
        1. Number of qubits
        2. Entanglement patterns
        3. Number of repetitions

        Returns:
            Dictionary with counts for each gate type
        """
        # Calculate minimum gates based on architecture
        min_hadamard = self.num_qubits * self.feature_map_hadamard_layers
        min_phase = 0
        min_cnot = 0
        min_ry = self.num_qubits * 2 * self.ansatz_reps

        # Add entanglement gates based on pattern
        if self.feature_map_entanglement == "full":
            connections = (self.num_qubits * (self.num_qubits - 1)) // 2
        elif self.feature_map_entanglement in ["linear", "circular"]:
            connections = self.num_qubits - 1
            if self.feature_map_entanglement == "circular":
                connections += 1

        min_phase += connections * self.feature_map_reps
        min_cnot += connections * 2 * self.feature_map_reps

        # Add ansatz entanglement gates
        if self.ansatz_entanglement == "full":
            min_cnot += (
                (self.num_qubits * (self.num_qubits - 1)) // 2
            ) * self.ansatz_reps
        else:  # linear or circular
            min_cnot += (self.num_qubits - 1) * self.ansatz_reps
            if self.ansatz_entanglement == "circular":
                min_cnot += self.ansatz_reps

        # Use target counts if specified, otherwise use calculated minimums
        hadamard = self.target_hadamard_gates or min_hadamard
        phase = self.target_phase_gates or min_phase
        cnot = self.target_cnot_gates or min_cnot
        ry = self.target_ry_gates or min_ry

        return {"h": hadamard, "p": phase, "cx": cnot, "ry": ry}


def create_quantum_network(
    config: Optional[QuantumArchitectureConfig] = None,
) -> EstimatorQNN:
    """Creates a quantum circuit with configurable architecture.

    This function builds a quantum circuit consisting of two main parts:
    1. Feature Map Circuit:
       - Encodes classical data into quantum states
       - Uses Pauli-Z evolution for feature interaction
       - Configurable entanglement pattern

    2. Variational Circuit (Ansatz):
       - Trainable rotation and entangling gates
       - Processes quantum features
       - Configurable depth and connectivity

    The implementation allows for:
    - Different entanglement patterns (full, linear, circular)
    - Variable circuit depth through repetitions
    - Optional gate count targets for matching specifications

    Args:
        config: Optional configuration for the quantum architecture.
               If None, uses default configuration.

    Returns:
        EstimatorQNN instance with the configured circuit

    Example Circuit Structure:
    ```
    Feature Map:          Ansatz:
    ┌───┐ ┌─┐   ┌───┐   ┌──────┐ ┌─┐   ┌──────┐
    ┤ H ├─┤P├───┤CX ├───┤ Ry(θ)├─┤M├───┤ Ry(θ)├
    ├───┤ └─┘┌──┴───┴──┐├──────┤ ├─┤┌──┴──────┴─┐
    ┤ H ├────┤CX      P├┤ Ry(θ)├─┤M├┤ Ry(θ)    ├
    └───┘    └─────────┘└──────┘ └─┘└───────────┘
    ```
    """
    if config is None:
        config = QuantumArchitectureConfig()
        logger.info("Using default quantum architecture configuration")

    logger.info("Creating quantum circuit with:")
    logger.info("- %d qubits", config.num_qubits)
    logger.info(
        "- Feature map: %s entanglement, %d reps",
        config.feature_map_entanglement,
        config.feature_map_reps,
    )
    logger.info(
        "- Ansatz: %s entanglement, %d reps",
        config.ansatz_entanglement,
        config.ansatz_reps,
    )

    # Create feature map circuit using pauli_feature_map function
    logger.debug("Building feature map circuit...")
    feature_map_circuit = pauli_feature_map(
        feature_dimension=config.num_qubits,
        paulis=["Z" * config.num_qubits],  # ZZ... interaction
        reps=config.feature_map_reps,
        insert_barriers=True,
    )

    # Create ansatz circuit using efficient_su2 function
    logger.debug("Building ansatz circuit...")
    ansatz_circuit = efficient_su2(
        num_qubits=config.num_qubits,
        reps=config.ansatz_reps,
        entanglement=config.ansatz_entanglement,
        insert_barriers=True,
    )

    # Combine circuits
    logger.debug("Combining feature map and ansatz circuits...")
    qc = QuantumCircuit(config.num_qubits)
    qc.compose(feature_map_circuit, inplace=True)
    qc.compose(ansatz_circuit, inplace=True)

    # Get parameters from circuits
    input_params = feature_map_circuit.parameters
    weight_params = ansatz_circuit.parameters

    # Track gate counts
    gate_counts = qc.count_ops()
    logger.info("Final quantum circuit structure:")
    logger.info("Gate counts: %s", gate_counts)
    logger.info("Parameters:")
    logger.info("- Input parameters: %d", len(input_params))
    logger.info("- Weight parameters: %d", len(weight_params))
    logger.info("- Total parameters: %d", len(input_params) + len(weight_params))

    # Create the Estimator QNN with V2 primitives
    logger.debug("Creating EstimatorQNN with StatevectorEstimator...")
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(input_params),
        weight_params=list(weight_params),
        estimator=StatevectorEstimator(),
    )

    return qnn


class ClassicalFeatureExtractor(nn.Module):
    """Classical CNN for feature extraction from 64x64 grayscale images.

    Architecture:
        1. Conv2D(1 -> 32) -> ReLU -> MaxPool
        2. Conv2D(32 -> 64) -> ReLU -> MaxPool -> Dropout
        3. Conv2D(64 -> 128) -> ReLU -> MaxPool -> Dropout
        4. Flatten -> Linear(2048 -> 128)

    Input shape: [batch_size, 1, 64, 64]
    Output shape: [batch_size, 128]
    """

    def __init__(self):
        super().__init__()

        # First conv block: 64x64x1 -> 32x32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Second conv block: 32x32x32 -> 16x16x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Third conv block: 16x16x64 -> 8x8x128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Final fully connected layer: 8*8*128 = 8192 -> 128
        self.fc = nn.Linear(8 * 8 * 128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input tensor of shape [batch_size, 1, 64, 64]

        Returns:
            Features tensor of shape [batch_size, 128]
        """
        # Apply convolution blocks
        x = self.conv1(x)  # -> [batch, 32, 32, 32]
        x = self.conv2(x)  # -> [batch, 64, 16, 16]
        x = self.conv3(x)  # -> [batch, 128, 8, 8]

        # Flatten and pass through FC layer
        x = x.view(x.size(0), -1)  # -> [batch, 8192]
        x = self.fc(x)  # -> [batch, 128]

        return x


class HybridQuantumModel(nn.Module):
    """Hybrid quantum-classical neural network for character recognition.

    This model combines classical and quantum processing:
    1. Classical CNN extracts features from images
    2. Features are reduced to quantum circuit input dimension
    3. Quantum circuit processes features
    4. Final classical layer maps to class probabilities

    Architecture Overview:
    ```
    Input Image    CNN Features    Quantum Features    Output
    [64x64x1] -> [128] ---------> [6] --------------> [36]
                     \\            /
                      -> reduce -/
    ```

    The quantum circuit can be configured for different:
    - Number of qubits
    - Entanglement patterns
    - Circuit depths
    - Gate configurations

    Example:
        >>> config = QuantumArchitectureConfig(num_qubits=8)
        >>> model = HybridQuantumModel(
        ...     quantum_config=config,
        ...     quantum_features=8,
        ...     classical_features=128
        ... )
    """

    def __init__(
        self,
        num_classes: int = 36,
        quantum_config: Optional[QuantumArchitectureConfig] = None,
        classical_features: int = 128,
        quantum_features: Optional[int] = None,
        debug_mode: bool = False,
    ):
        """Initialize the hybrid quantum-classical model.

        Args:
            num_classes: Number of output classes (default 36 for 0-9,A-Z)
            quantum_config: Configuration for quantum circuit architecture
            classical_features: Number of features from CNN
            quantum_features: Number of features for quantum processing
            debug_mode: Whether to log feature statistics during forward pass
        """
        super().__init__()

        if quantum_config is None:
            quantum_config = QuantumArchitectureConfig()

        quantum_features = quantum_features or quantum_config.num_qubits

        if quantum_features != quantum_config.num_qubits:
            raise ValueError(
                f"quantum_features ({quantum_features}) must match "
                f"num_qubits in quantum_config ({quantum_config.num_qubits})"
            )

        # Log configuration details
        logger.info("Initializing HybridQuantumModel:")
        logger.info("- Classes: %d", num_classes)
        logger.info("- Classical features: %d", classical_features)
        logger.info("- Quantum features: %d", quantum_features)
        logger.info(
            "- Feature map: %s entanglement, %d reps",
            quantum_config.feature_map_entanglement,
            quantum_config.feature_map_reps,
        )
        logger.info(
            "- Ansatz: %s entanglement, %d reps",
            quantum_config.ansatz_entanglement,
            quantum_config.ansatz_reps,
        )

        # Log expected gate counts
        gate_counts = quantum_config.get_total_gates()
        logger.info("Expected quantum gate counts:")
        for gate, count in gate_counts.items():
            logger.info("- %s gates: %d", gate.upper(), count)

        # Store configuration
        self.quantum_config = quantum_config
        self.debug_mode = debug_mode

        # Initialize network components
        logger.debug("Creating classical CNN feature extractor...")
        self.cnn = ClassicalFeatureExtractor()

        logger.debug("Adding feature reduction layer...")
        self.feature_reduction = nn.Linear(classical_features, quantum_features)

        # Create quantum circuit
        logger.debug("Building quantum circuit...")
        qnn = create_quantum_network(quantum_config)
        self.qnn_torch = TorchConnector(qnn)

        logger.debug("Adding final classification layer...")
        self.classifier = nn.Linear(1, num_classes)

        logger.info("Model initialization complete")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model.

        Processing steps:
        1. Extract classical features using CNN
        2. Reduce features to quantum dimension
        3. Process features with quantum circuit
        4. Map quantum output to class probabilities

        Args:
            x: Input tensor of shape [batch_size, 1, 64, 64]

        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]

        # Classical feature extraction
        features = self.cnn(x)
        if self.debug_mode:
            logger.debug(
                "Classical features: shape=%s, range=[%.3f, %.3f]",
                features.shape,
                features.min().item(),
                features.max().item(),
            )

        # Reduce to quantum features
        quantum_features = self.feature_reduction(features)
        if self.debug_mode:
            logger.debug(
                "Quantum input: shape=%s, range=[%.3f, %.3f]",
                quantum_features.shape,
                quantum_features.min().item(),
                quantum_features.max().item(),
            )

        # Quantum processing
        q_out = self.qnn_torch(quantum_features)
        if self.debug_mode:
            logger.debug(
                "Quantum output: shape=%s, range=[%.3f, %.3f]",
                q_out.shape,
                q_out.min().item(),
                q_out.max().item(),
            )

        # Final classification
        logits = self.classifier(q_out)
        if self.debug_mode:
            logger.debug(
                "Final logits: shape=%s, range=[%.3f, %.3f]",
                logits.shape,
                logits.min().item(),
                logits.max().item(),
            )

        return logits

    def save_config(self, path: str) -> None:
        """Save quantum configuration to file for reproducibility."""
        self.quantum_config.save(path)

    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> "HybridQuantumModel":
        """Create model instance from saved configuration.

        Args:
            config_path: Path to saved quantum configuration
            **kwargs: Additional arguments for model initialization

        Returns:
            Initialized model with loaded configuration
        """
        config = QuantumArchitectureConfig.load(config_path)
        return cls(quantum_config=config, **kwargs)


###############################################################################
# Training & Evaluation Functions
###############################################################################
# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def train_model_full(model, loader, optimizer, criterion, device, epochs=2):
    """
    A standard training loop that iterates over ALL batches in loader for each epoch.

    Args:
        model: The model to train
        loader: DataLoader with training data
        optimizer: The optimizer to use
        criterion: Loss function
        device: Device to run on (CPU or CUDA)
        epochs: Number of epochs to train for
    """
    model.to(device)
    logger.info(
        "Starting training for %s epochs (%s batches per epoch)", epochs, len(loader)
    )

    total_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_start_time = time.time()

        logger.info(
            "Epoch %s/%s - processing %s batches", epoch + 1, epochs, len(loader)
        )

        for i, (images, labels) in enumerate(loader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            batch_count += 1

            # Log progress periodically
            if (i + 1) % max(1, len(loader) // 10) == 0:  # Log ~10 times per epoch
                batch_time = time.time() - batch_start
                logger.info(
                    "  Batch %s/%s - Loss: %.4f, Time: %.2fs",
                    i + 1,
                    len(loader),
                    loss.item(),
                    batch_time,
                )

        # Epoch summary
        avg_loss = running_loss / batch_count
        epoch_time = time.time() - epoch_start_time
        logger.info(
            "Epoch %s completed - Avg Loss: %.4f, Time: %.2fs",
            epoch + 1,
            avg_loss,
            epoch_time,
        )

    # Training summary
    total_time = time.time() - total_start_time
    logger.info("Training completed in %.2fs", total_time)


# pylint: disable=too-many-locals
def evaluate_model(model, loader, device):
    """
    Full pass over the test loader to measure accuracy.

    Args:
        model: The model to evaluate
        loader: DataLoader with test data
        device: Device to run on (CPU or CUDA)

    Returns:
        accuracy: The model's accuracy as a percentage
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    logger.info("Evaluating model on %s test batches", len(loader))
    start_time = time.time()

    confusion_matrix = torch.zeros(36, 36, dtype=torch.int)

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            preds = logits.argmax(dim=1)

            # Update statistics
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Log progress periodically
            if (i + 1) % max(
                1, len(loader) // 5
            ) == 0:  # Log ~5 times during evaluation
                logger.info("  Processed %s/%s test batches", i + 1, len(loader))

    # Calculate accuracy
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    eval_time = time.time() - start_time

    logger.info(
        "Evaluation completed in %.2fs - Accuracy: %.2f%% (%s/%s)",
        eval_time,
        accuracy,
        correct,
        total,
    )

    # Log per-class accuracies (diagonal of confusion matrix)
    class_correct = confusion_matrix.diag().tolist()
    class_total = confusion_matrix.sum(1).tolist()
    class_accuracy = [
        100.0 * c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)
    ]

    # Log some class accuracies (digits and some letters)
    digits_acc = class_accuracy[:10]  # 0-9
    letters_acc = class_accuracy[10:36]  # A-Z

    logger.debug("Digit accuracies: %s", digits_acc)
    logger.debug("Letter accuracies: %s", letters_acc)
    logger.info("Avg digit accuracy: %.2f%%", sum(digits_acc) / len(digits_acc))
    logger.info("Avg letter accuracy: %.2f%%", sum(letters_acc) / len(letters_acc))

    return accuracy


###############################################################################
# Main training function
###############################################################################
# pylint: disable=too-many-locals
def train_and_evaluate(batch_size=16, epochs=2, learning_rate=1e-3, log_level=None):
    """
    Complete pipeline for training and evaluating the hybrid quantum-classical model.

    Args:
        batch_size: Size of mini-batches for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the Adam optimizer
        log_level: Optional logging level to set for this module

    Returns:
        Tuple[nn.Module, float]: The trained model and final accuracy
    """
    # Set log level if provided
    if log_level is not None:
        logger.setLevel(log_level)

    # Determine device (CUDA/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Log device details
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info(
            "CUDA memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Load and prepare the dataset
    train_loader, test_loader, num_classes = load_and_prepare_dataset(batch_size)

    # Initialize the model
    model = HybridQuantumModel(num_classes=num_classes)

    # Log model summary (manually since no built-in summary)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model initialized with %s parameters (%s trainable)",
        total_params,
        trainable_params,
    )

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    logger.info("Using Adam optimizer with learning rate %s", learning_rate)

    # Train the model
    train_model_full(model, train_loader, optimizer, criterion, device, epochs=epochs)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader, device)

    # Show example predictions
    logger.info("Sample predictions:")
    example_images, example_labels = next(iter(test_loader))
    example_images, example_labels = example_images.to(device), example_labels.to(
        device
    )
    with torch.no_grad():
        example_logits = model(example_images)
        example_preds = example_logits.argmax(dim=1)

    # Map numeric labels back to characters for better readability
    def label_to_char(label):
        if 0 <= label <= 9:
            return str(label)
        if 10 <= label <= 35:
            return chr(ord("A") + label - 10)
        return "?"

    # Show first few predictions
    pred_chars = [label_to_char(l) for l in example_preds[:8].cpu().tolist()]
    true_chars = [label_to_char(l) for l in example_labels[:8].cpu().tolist()]

    logger.info("  Predicted: %s", pred_chars)
    logger.info("  Actual:    %s", true_chars)

    return model, accuracy
