# QNet - Hybrid Quantum-Classical Neural Network

This module implements a hybrid quantum-classical neural network for character recognition, specifically designed for license plate characters (0-9, A-Z).

## Model Architecture

The `HybridModel` combines classical and quantum computing components:

1. **Classical CNN Feature Extractor**:
   - 3 Conv2D + MaxPool2D blocks (32, 64, 128 filters)
   - Processes 64x64 input images to 8x8 feature maps
   - Dense layers with dropout for dimensionality reduction
   - Final tanh activation to produce quantum circuit inputs

2. **Quantum Circuit**:
   - ZZFeatureMap for encoding classical features into quantum states
   - RealAmplitudes variational ansatz with configurable depth
   - SamplerQNN with gradient support
   - Torch connector for PyTorch integration

3. **Classical Output Layer**:
   - Maps quantum measurement outcomes to class probabilities

## CLI Usage

The module provides a command-line interface through the `ipsem2025-train` command:

```bash
# Train a model with default parameters
ipsem2025-train train

# Train with custom parameters
ipsem2025-train train \
    --n-qubits 4 \
    --ansatz-reps 2 \
    --epochs 10 \
    --batch-size 64 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --learning-rate 0.001 \
    --dataset-type emnist \
    --dataset-path data \
    --model-save-path model.pt \
    --stats-file training_stats.csv \
    --test  # Run evaluation after training
    --log-file train.log \
    --verbose

# Test a trained model
ipsem2025-train test \
    --model-path model.pt \
    --dataset-type emnist \
    --dataset-path data \
    --batch-size 32 \
    --verbose
```

### CLI Options

Training options:
- `--n-qubits, -q`: Number of qubits (default: 2)
- `--ansatz-reps, -r`: Depth of quantum circuit (default: 1)
- `--epochs, -e`: Training epochs (default: 3)
- `--batch-size, -b`: Batch size (default: 32)
- `--train-ratio`: Ratio of data to use for training (default: 0.7)
- `--val-ratio`: Ratio of data to use for validation (default: 0.15)
- `--learning-rate, -l`: Learning rate (default: 0.001)
- `--dataset-type, -d`: Dataset type (emnist, mnist, or custom)
- `--dataset-path, -p`: Path to dataset (default: "data")
- `--model-save-path, -m`: Path to save trained model
- `--stats-file, -s`: Path to save training statistics CSV (default: "training_stats.csv")
- `--test, -t`: Run evaluation after training
- `--log-file`: Path to save log output
- `--verbose, -v`: Enable verbose output

Testing options:
- `--model-path, -m`: Path to saved model (required)
- `--dataset-type, -d`: Dataset type (emnist, mnist, or custom)
- `--dataset-path, -p`: Path to dataset
- `--batch-size, -b`: Batch size for testing (default: 32)
- `--verbose, -v`: Enable verbose output

## Library Usage

The module can also be used as a library for custom training pipelines:

```python
from ipsem2025_license_plate.qnet.model import HybridModel
from ipsem2025_license_plate.qnet.train import train_hybrid_model
from ipsem2025_license_plate.datasets.emnist import EMNISTDataset

# Train a model
result = train_hybrid_model(
    n_qubits=4,
    ansatz_reps=2,
    epochs=10,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    learning_rate=0.001,
    dataset_type="emnist",
    dataset_path="data",
    model_save_path="model.pt",
    stats_file="training_stats.csv",
    log_file="train.log",
    run_test=True,
    verbose=True
)

# Access results
model = result['model']
final_stats = result['final_stats']
training_history = result['training_history']
test_metrics = result['test_metrics']

# Use the model directly
dataset = EMNISTDataset(root="data")
model = HybridModel(
    n_qubits=4, 
    ansatz_reps=2, 
    num_classes=36,
    input_channels=1  # grayscale images
)
model.load_state_dict(torch.load("model.pt"))

# Make predictions
image, _ = dataset[0]
output = model(image.unsqueeze(0))
prediction = output.argmax(dim=1).item()

# Get model information
model_info = model.get_model_info()
print(f"Model has {model_info['n_qubits']} qubits")
print(f"Circuit depth: {model_info['circuit_depth']}")
print(f"Total parameters: {model_info['total_params']}")
```

### Training Statistics

The training process records detailed statistics in CSV format:
- Loss and accuracy per epoch
- Training time and memory usage
- Hardware information
- Model parameters

## Supported Datasets

The module works with multiple dataset types:
- **EMNIST**: Extended MNIST dataset with letters and digits
- **MNIST**: Classic handwritten digit dataset
- **Custom**: User-provided image datasets

See the [Datasets Module](../datasets/README.md) for more information on dataset management.
