"""Tests for quantum architecture configuration and gate counts."""

import os

import pytest
import torch

from ipsem2025_license_plate.qnet.hybrid_net import (
    HybridQuantumModel,
    QuantumArchitectureConfig,
)
from ipsem2025_license_plate.utils.logging_utils import get_logger

logger = get_logger(__name__)


def test_default_configuration():
    """Test default quantum architecture configuration."""
    config = QuantumArchitectureConfig()

    assert config.num_qubits == 5
    assert config.feature_map_reps == 2
    assert config.feature_map_entanglement == "full"
    assert config.ansatz_reps == 1
    assert config.ansatz_entanglement == "full"

    # Verify target gate counts for 5 qubits
    assert config.target_hadamard_gates == 10
    assert config.target_phase_gates == 35
    assert config.target_cnot_gates == 50
    assert config.target_ry_gates == 10
    assert config.target_ansatz_cnot_gates == 4


def test_linear_entanglement_config():
    """Test configuration with linear entanglement pattern."""
    config = QuantumArchitectureConfig(
        num_qubits=8,
        feature_map_entanglement="linear",
        ansatz_entanglement="linear",
        feature_map_reps=3,
        ansatz_reps=2,
        # No specific gate targets
        target_hadamard_gates=None,
        target_phase_gates=None,
        target_cnot_gates=None,
        target_ry_gates=None,
        target_ansatz_cnot_gates=None,
    )

    # Calculate expected gate counts
    gate_counts = config.get_total_gates()

    # Linear entanglement should have n-1 connections per layer
    assert config.num_qubits == 8
    assert gate_counts["h"] == 8 * config.feature_map_hadamard_layers
    assert gate_counts["ry"] == 8 * 2 * config.ansatz_reps  # 2 Ry per qubit per rep


def test_circular_entanglement_config():
    """Test configuration with circular entanglement pattern."""
    config = QuantumArchitectureConfig(
        num_qubits=4,
        feature_map_entanglement="circular",
        ansatz_entanglement="circular",
        feature_map_reps=2,
        ansatz_reps=3,
        # Override with specific gate targets for 4 qubits
        target_hadamard_gates=8,  # 4 qubits * 2 layers
        target_phase_gates=32,  # Adjusted for 4 qubits
        target_cnot_gates=40,  # Adjusted for 4 qubits
        target_ry_gates=8,  # 4 qubits * 2 reps
    )

    # Calculate expected gate counts
    gate_counts = config.get_total_gates()

    # Circular entanglement should have n connections per layer
    assert config.num_qubits == 4
    assert gate_counts["h"] == 8  # 4 qubits * 2 layers
    assert gate_counts["p"] == 32  # Adjusted for 4 qubits
    assert gate_counts["cx"] == 40  # Adjusted for 4 qubits
    assert gate_counts["ry"] == 8  # 4 qubits * 2 reps


def test_config_save_load(tmp_path):
    """Test saving and loading configurations."""
    original_config = QuantumArchitectureConfig(
        num_qubits=10,
        feature_map_reps=4,
        feature_map_entanglement="linear",
        ansatz_reps=2,
        ansatz_entanglement="circular",
    )

    # Save configuration
    config_path = os.path.join(tmp_path, "quantum_config.json")
    original_config.save(config_path)

    # Load configuration
    loaded_config = QuantumArchitectureConfig.load(config_path)

    # Verify all attributes match
    assert loaded_config.num_qubits == original_config.num_qubits
    assert loaded_config.feature_map_reps == original_config.feature_map_reps
    assert (
        loaded_config.feature_map_entanglement
        == original_config.feature_map_entanglement
    )
    assert loaded_config.ansatz_reps == original_config.ansatz_reps
    assert loaded_config.ansatz_entanglement == original_config.ansatz_entanglement


def test_model_with_custom_config():
    """Test model creation with custom quantum configuration."""
    config = QuantumArchitectureConfig(
        num_qubits=8,
        feature_map_entanglement="linear",
        ansatz_entanglement="circular",
        feature_map_reps=2,
        ansatz_reps=3,
    )

    model = HybridQuantumModel(
        quantum_config=config, quantum_features=8, classical_features=128
    )

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64)  # Test input
    output = model(x)

    assert output.shape == (batch_size, 36)  # 36 classes


def test_invalid_configurations():
    """Test that invalid configurations are rejected."""
    # Test invalid number of qubits
    with pytest.raises(ValueError):
        QuantumArchitectureConfig(num_qubits=0)

    # Test invalid repetitions
    with pytest.raises(ValueError):
        QuantumArchitectureConfig(feature_map_reps=0)

    with pytest.raises(ValueError):
        QuantumArchitectureConfig(ansatz_reps=0)

    # Test mismatched features and qubits
    config = QuantumArchitectureConfig(num_qubits=6)
    with pytest.raises(ValueError):
        HybridQuantumModel(quantum_config=config, quantum_features=8)
