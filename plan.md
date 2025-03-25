# Implementation Plan for Enhanced CLI and Dataset Flexibility

## 1. Dataset Management CLI Commands
### Download Dataset Command
- Add `dataset download` subcommand
- Options:
  - `--dataset-name`: Choose which dataset to download (EMNIST, custom)
  - `--output-dir`: Where to save the dataset
  - `--force`: Force re-download even if exists

### Dataset Preview Command
- Add `dataset preview` subcommand
- Options:
  - `--num-samples`: Number of samples to show (default: 8)
  - `--save-plot`: Save preview to file instead of displaying
  - `--dataset-path`: Path to custom dataset
  - Features:
    - Display sample images with labels
    - Show class distribution
    - Show image dimensions and statistics

### Dataset Info Command
- Add `dataset info` subcommand
- Show:
  - Total samples
  - Class distribution
  - Image dimensions
  - File format
  - Dataset structure

### Local Dataset Loading
- Add support for loading local datasets:
  - `--dataset-path`: Path to local dataset directory
  - Support common formats:
    - Directory of images organized by class folders
    - CSV files with paths and labels
    - HDF5/NPZ archives
  - Auto-detect dataset structure
  - Validate dataset format and structure
  - Options for specifying class mappings
  - Support for custom label encodings

## 2. Training Enhancements
### Partial Dataset Training
- Add training options:
  - `--train-size`: Percentage or number of samples to use for training (e.g., "10%", "1000")
  - `--val-size`: Validation set size
  - `--stratify`: Whether to maintain class distribution in partial sets
  - `--random-seed`: For reproducibility

### Checkpointing and Resume
- Add checkpointing:
  - `--checkpoint-dir`: Directory to save checkpoints
  - `--checkpoint-freq`: How often to save (epochs)
  - `--resume`: Resume from checkpoint file

### Model Saving and Loading
- Add model persistence:
  - `--save-model`: Save model after training
  - `--model-path`: Path to save/load model
  - `--save-format`: Format to save model in (pytorch, onnx)
  - Save additional metadata:
    - Training configuration
    - Dataset information
    - Class mappings
    - Performance metrics
  - Export options:
    - Full model with weights
    - Architecture only
    - Optimized/quantized versions
  - Version control for saved models

## 3. Network Flexibility

### Quantum-Classical Hybrid Architecture

#### Classical CNN Component
1. Convolutional Layers:
   ```
   - Conv2D (32 filters, 3x3 kernel, ReLU)
   - MaxPooling2D (2x2)
   - Conv2D (64 filters, 3x3 kernel, ReLU)
   - MaxPooling2D (2x2)
   - Conv2D (128 filters, 3x3 kernel, ReLU)
   - MaxPooling2D (2x2)
   - Flatten
   - Dense (128 units, ReLU)
   - Dropout (0.5)
   ```

#### Quantum Component
1. Feature Encoding (ZZFeatureMap):
   - Encodes 6 classical features into quantum states
   - Gate Configuration:
     - 12 Hadamard gates (superposition)
     - 42 Phase gates (phase shifts)
     - 59 CNOT gates (entanglement)
   - Total: 113 gates
   - Purpose: Second-order Pauli-Z evolution circuit for feature interaction

2. Quantum Processing (RealAmplitudes):
   - Processes encoded quantum states
   - Gate Configuration:
     - 12 Ry rotation gates
     - 5 CNOT gates
   - Maintains real amplitudes for classical post-processing

3. Output Layer:
   - Dense (36 units, softmax)
   - Maps quantum output to class probabilities

#### Architecture Benefits
1. Data Efficiency:
   - High accuracy with limited datasets
   - Particularly valuable for scarce data scenarios

2. Enhanced Pattern Recognition:
   - Quantum component improves generalization
   - Captures complex patterns beyond classical capabilities

3. Computational Advantages:
   - Potential reduction in computational complexity
   - Improved time and resource efficiency for specific tasks

#### Implementation Requirements
1. CNN Feature Extraction:
   - Implement configurable CNN layers
   - Support dynamic input dimensions
   - Proper feature normalization

2. Quantum Circuit Implementation:
   - ZZFeatureMap for 5-qubit encoding
   - RealAmplitudes circuit with specific gate counts
   - Efficient classical-to-quantum data transfer

3. Hybrid Integration:
   - Seamless classical-quantum transition
   - Proper dimensionality matching
   - Efficient backpropagation through both components

4. Performance Optimization:
   - Gate-level optimization
   - Batch processing efficiency
   - Memory management for quantum states

### Network Architecture Adaptation
1. Make networks dimension-agnostic:
   - Dynamically configure input layers based on image dimensions
   - Adjust CNN architecture based on input size
   - Configure output layer based on number of classes

2. Add configuration options:
   - Model architecture selection
   - Layer sizes
   - Activation functions

## 4. Configuration System
1. Add configuration file support:
   - YAML/JSON config files for complex setups
   - Command line overrides for quick changes
   - Default configurations

2. Configurable parameters:
   - Dataset paths and options
   - Model architecture
   - Training parameters
   - Logging settings

## 5. Project Structure Changes
### New Files to Create:
- `src/ipsem2025_license_plate/datasets/`
  - `__init__.py`
  - `base.py` (dataset interfaces)
  - `emnist.py` (current dataset implementation)
  - `custom.py` (custom dataset implementation)
  - `utils.py` (shared dataset utilities)

- `src/ipsem2025_license_plate/models/`
  - `__init__.py`
  - `base.py` (model interfaces)
  - `hybrid_net.py` (moved from qnet/)
  - `classical_net.py` (classical-only variant)
  - `config.py` (model configuration)

### File Structure Cleanup
1. CLI Organization (`cli.py`):
   - Move all CLI-related code to cli.py
   - Organize commands into logical groups:
     - Dataset commands (download, preview, info)
     - Training commands (train, evaluate)
     - Model commands (save, load, export)
   - Keep command implementations in cli.py
   - Use submodules for complex command logic

2. Main Module (`ipsem2025_license_plate.py`):
   - Convert to clean API entry point
   - Export public interfaces only
   - Provide clean imports for users
   - Example structure:
     ```python
     # Public API exports
     from .datasets import EMNISTDataset, CustomDataset
     from .models import HybridQuantumModel
     from .training import train_model, evaluate_model
     
     __all__ = [
         'EMNISTDataset',
         'CustomDataset',
         'HybridQuantumModel',
         'train_model',
         'evaluate_model'
     ]
     ```

3. Module Organization:
   - Move qnet/ contents to appropriate new locations:
     - dataset_utils.py → datasets/
     - hybrid_net.py → models/
   - Refactor utils/ for clarity:
     - cli_utils.py → cli/utils.py
     - logging_utils.py → core/logging.py

4. Clean Import Structure:
   - Use relative imports within the package
   - Avoid circular dependencies
   - Group related functionality
   - Provide type hints for public APIs

### Files to Modify:
- `cli.py`: 
  - Restructure as main CLI entry point
  - Add all command implementations
  - Use command groups for organization

- `ipsem2025_license_plate.py`:
  - Convert to clean API entry point
  - Remove direct functionality
  - Export public interfaces

- `dataset_utils.py`: 
  - Move to datasets/
  - Split into logical components

- `hybrid_net.py`:
  - Move to models/
  - Make more flexible

## 6. CLI Command Organization
### Entry Points in pyproject.toml
1. Main Command Group:
   ```toml
   [tool.poetry.scripts]
   ipsem2025 = "ipsem2025_license_plate.cli:main"
   ```

2. Specialized Commands:
   - Dataset Management:
     - `ipsem2025-dataset` = "ipsem2025_license_plate.cli_dataset:main"
     - Functions: download, preview, info, validate
   
   - Training:
     - `ipsem2025-train` = "ipsem2025_license_plate.cli_train:main"
     - Functions: train, resume, evaluate
   
   - Model Management:
     - `ipsem2025-model` = "ipsem2025_license_plate.cli_model:main"
     - Functions: save, load, export, optimize

### CLI File Structure
- Main package directory:
  - `cli.py` - Main unified CLI with subcommands
  - `cli_dataset.py` - Dataset management commands
  - `cli_train.py` - Training related commands
  - `cli_model.py` - Model management commands
  - `cli_utils.py` - Shared CLI utilities (moved from utils/)

### Command Usage Examples
1. Dataset Operations:
   ```bash
   ipsem2025-dataset download --dataset-name emnist
   ipsem2025-dataset preview --dataset-path ./my_dataset
   ipsem2025-dataset info --dataset-path ./my_dataset
   ```

2. Training Operations:
   ```bash
   ipsem2025-train start --dataset-path ./my_dataset
   ipsem2025-train resume --checkpoint ./checkpoints/last.pt
   ipsem2025-train evaluate --model-path ./models/mymodel.pt
   ```

3. Model Operations:
   ```bash
   ipsem2025-model save --output ./models/mymodel.pt
   ipsem2025-model export --format onnx --output ./models/mymodel.onnx
   ipsem2025-model optimize --target mobile
   ```

4. Main CLI (combines all):
   ```bash
   ipsem2025 dataset download
   ipsem2025 train start
   ipsem2025 model export
   ```

### Implementation Steps
1. Move cli_utils.py to main package directory
2. Create individual CLI command files
3. Update pyproject.toml with new entry points
4. Implement command-specific argument parsing
5. Add command documentation and help texts
6. Create integration tests for each command

## Implementation Order
1. Dataset abstraction layer & restructuring
2. Network flexibility changes
3. CLI command additions
4. Configuration system
5. Training enhancements
6. Documentation updates

## Testing Requirements
1. Unit tests for:
   - Dataset loading and preprocessing
   - Model configuration
   - CLI commands
   - Configuration parsing

2. Integration tests for:
   - Full training pipeline
   - Dataset conversion
   - Checkpoint/resume functionality

## Documentation Updates
1. Update README.md with new features
2. Add detailed CLI documentation
3. Add configuration file examples
4. Add custom dataset guide
5. Update developer documentation