# QNet - Hybrid Quantum-Classical Neural Network

This module implements a hybrid quantum-classical neural network for character recognition, specifically designed for license plate characters (0-9, A-Z).

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
    --learning-rate 0.001 \
    --dataset-type emnist \
    --model-save-path model.pt \
    --stats-file training_stats.csv \
    --test  # Run evaluation after training

# Test a trained model
ipsem2025-train test \
    --model-path model.pt \
    --dataset-type emnist \
    --batch-size 32
```

### CLI Options

Training options:
- `--n-qubits, -q`: Number of qubits (default: 2)
- `--ansatz-reps, -r`: Depth of quantum circuit (default: 1)
- `--epochs, -e`: Training epochs (default: 3)
- `--batch-size, -b`: Batch size (default: 32)
- `--learning-rate, -l`: Learning rate (default: 0.001)
- `--dataset-type, -d`: Dataset type (emnist, mnist, or custom)
- `--dataset-path, -p`: Path to dataset
- `--model-save-path, -m`: Path to save trained model
- `--stats-file, -s`: Path to save training statistics CSV
- `--test, -t`: Run evaluation after training
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
    learning_rate=0.001,
    dataset_type="emnist",
    dataset_path="data",
    model_save_path="model.pt",
    stats_file="training_stats.csv",
    run_test=True
)

# Access results
model = result['model']
final_stats = result['final_stats']
training_history = result['training_history']
test_metrics = result['test_metrics']

# Use the model directly
dataset = EMNISTDataset(root="data")
model = HybridModel(n_qubits=4, ansatz_reps=2, num_classes=36)
model.load_state_dict(torch.load("model.pt"))

# Make predictions
image, _ = dataset[0]
output = model(image.unsqueeze(0))
prediction = output.argmax(dim=1).item()
```

### Training Statistics

The training process records detailed statistics in CSV format:
- Loss and accuracy per epoch
- Training time and memory usage
- Hardware information
- Model parameters

Example stats file content:
```csv
timestamp,epoch,loss,accuracy,samples_processed,epoch_time,total_time,learning_rate,cpu_memory_mb,gpu_memory_mb
2024-02-20T10:15:30,1,2.1543,45.67,1000,120.5,120.5,0.001,1024.5,512.3
2024-02-20T10:17:30,2,1.8765,62.34,1000,119.8,240.3,0.001,1025.1,512.3
```
