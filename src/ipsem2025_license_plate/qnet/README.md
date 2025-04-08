# QNet - Hybrid Quantum-Classical Neural Network

This module implements a hybrid quantum-classical neural network for character recognition, specifically designed for license plate characters (0-9, A-Z).

## Model Architecture

The `HybridModel` combines classical and quantum computing components:

1. **Classical CNN Feature Extractor**:
   - Convolutional layers with pooling and dropout
   - Dense layers for dimensionality reduction
   - Transforms input images into quantum circuit inputs

2. **Quantum Circuit**:
   - ZZFeatureMap for encoding classical features into quantum states
   - RealAmplitudes variational ansatz with configurable depth
   - SamplerQNN implementation
   - Transpiled circuits for better GPU performance
   - Torch connector for PyTorch integration
   - GPU-accelerated circuit simulation (via qiskit-aer-gpu)

3. **Classical Output Layer**:
   - Maps quantum measurement outcomes to class probabilities

## Quantum Neural Network

The model uses a SamplerQNN from Qiskit Machine Learning:

- Based on quantum sampling primitives
- Returns 2^n_qubits values per input (where n_qubits is the number of qubits)
- Provides rich information about quantum state probabilities
- Supports GPU acceleration for faster training

## GPU-Accelerated Quantum Simulation

The model supports GPU-accelerated quantum circuit simulation via the qiskit-aer-gpu package:

- **Faster Training**: Significantly speeds up quantum circuit simulations, especially for larger circuits
- **Automatic Detection**: Detects available GPUs and uses them automatically
- **Configurable**: Can be enabled/disabled via CLI option
- **Enhanced Performance Tracking**: Detailed GPU memory tracking and device placement
- **Circuit Transpilation**: Circuits are transpiled to the backend for better GPU performance
- **Graceful Fallback**: Falls back to CPU if GPU is not available

To use GPU acceleration, ensure you have the `qiskit-aer-gpu` package installed and a CUDA-compatible GPU.

## CLI Usage

The module provides a command-line interface through the `ipsem2025-train` command:

```bash
# Train a model with default parameters
ipsem2025-train train

# Train with maximum GPU performance
ipsem2025-train train \
    --n-qubits 6 \
    --ansatz-reps 3 \
    --feature-map-reps 2 \
    --epochs 10 \
    --batch-size 128 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --learning-rate 0.001 \
    --dataset-type emnist \
    --dataset-path data \
    --output-dir runs \
    --save-intermediate \
    --test \
    --preload-data \
    --num-workers 8 \
    --verbose \
    --use-gpu-for-qnn

# Disable GPU acceleration (use CPU only)
ipsem2025-train train --no-gpu-for-qnn

# Disable intermediate model saving
ipsem2025-train train --no-save-intermediate

# Test a trained model
ipsem2025-train test \
    --model-path model.pt \
    --dataset-type emnist \
    --dataset-path data \
    --batch-size 128 \
    --verbose
```

### CLI Options

Training options:
- `--n-qubits, -q`: Number of qubits (default: 6)
- `--ansatz-reps, -r`: Depth of quantum ansatz circuit (default: 2)
- `--feature-map-reps`: Depth of quantum feature map circuit (default: 1)
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
- `--use-gpu-for-qnn/--no-gpu-for-qnn`: Enable/disable GPU acceleration for quantum circuit simulation (default: enabled)
- `--save-intermediate/--no-save-intermediate`: Save intermediate models after each epoch (default: enabled)
- `--output-dir, -o`: Base directory for run outputs (default: "runs")
- `--preload-data`: Preload dataset into memory for faster training (default: False)
- `--num-workers`: Number of dataloader worker processes (default: auto-detect)

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
import multiprocessing
from pathlib import Path

# Calculate optimal number of worker processes (half of available cores)
num_workers = max(1, multiprocessing.cpu_count() // 2)

# Create a run directory
run_dir = Path("runs/my_training_run")
run_dir.mkdir(parents=True, exist_ok=True)

# Train a model with maximum GPU performance
result = train_hybrid_model(
    n_qubits=6,
    ansatz_reps=3,
    feature_map_reps=2,  # Deeper feature map for better encoding
    epochs=20,
    batch_size=128,      # Larger batch size for better GPU utilization
    train_ratio=0.7,
    val_ratio=0.15,
    learning_rate=0.001,
    dataset_type="emnist",
    dataset_path="data",
    model_save_path=str(run_dir / "model.pt"),
    stats_file=run_dir / "training_stats.csv",
    run_dir=run_dir,     # All outputs including logs will go here
    run_test=True,
    verbose=True,
    use_gpu_for_qnn=True,  # Enable GPU acceleration for quantum circuits
    save_intermediate=True, 
    preload_data=True,     # Preload entire dataset into memory for faster access
    num_workers=num_workers # Use optimal number of worker processes
)

# Access results
model = result['model']
final_stats = result['final_stats']
training_history = result['training_history']
test_metrics = result['test_metrics']
run_dir = result['run_dir']  # Directory where all run outputs are saved

# Use the model directly
dataset = EMNISTDataset(root="data", preload_to_memory=True)  # Preload data for faster inference
model = HybridModel(
    n_qubits=6, 
    ansatz_reps=3,
    feature_map_reps=2,
    num_classes=36,
    input_channels=1,
    use_gpu=True,
    device=torch.device("cuda")
)
model.load_state_dict(torch.load("model.pt"))

# Check parameter device placement
for name, param in model.named_parameters():
    print(f"Parameter '{name}' is on device: {param.device}")

# Make predictions
image, _ = dataset[0]
output = model(image.unsqueeze(0))
prediction = output.argmax(dim=1).item()

# Get model information
model_info = model.get_model_info()
print(f"Model has {model_info['n_qubits']} qubits")
print(f"Circuit depth: {model_info['circuit_depth']}")
print(f"Total parameters: {model_info['total_params']}")
print(f"Using GPU for quantum simulation: {model_info['using_gpu_quantum']}")
```

### Performance Optimization

For maximum performance on GPU:

1. **Preload Data**: Use `--preload-data` to load the entire dataset into memory
2. **Set Workers Optimally**: Set `--num-workers` to half the number of CPU cores
3. **Increase Batch Size**: Use larger batch sizes (128-256) for better GPU utilization
4. **Use GPU Acceleration**: Ensure `--use-gpu-for-qnn` is enabled
5. **Pin Memory**: This is automatically enabled for GPU training
6. **Optimize Circuit Depth**: Balance between expressive power (deeper circuits) and training time

### Training Statistics

The training process records detailed statistics in CSV format:
- Loss and accuracy per epoch
- Training time and memory usage
- GPU memory usage tracking
- Hardware information
- Model parameters

### Example Training Statistics CSV

Here's an example of the statistics CSV file generated during training:

```csv
timestamp,epoch,train_loss,train_accuracy,val_loss,val_accuracy,samples_processed,epoch_time,total_time,learning_rate,cpu_memory_mb,gpu_memory_mb,gpu_peak_memory_mb,batch_size,device,cpu_model,python_version,torch_version,cuda_version,total_memory,cpu_count,cpu_threads,gpu_model,gpu_memory
2025-03-28T20:47:58.651611,1,3.348637762759262,6.217598908594816,3.286354132574653,6.647339699863575,58640,809.327287197113,1007.9645798206329,0.001,1277.4140625,3584.75,4096.5,32,cuda:0,x86_64,3.12.8,2.6.0+cu124,12.0,62.7GB,6,12,NVIDIA A100,40GB
```

Key statistics tracked:
- **timestamp**: Date and time when the epoch completed
- **epoch**: Training epoch number
- **train_loss/val_loss**: Loss on training and validation sets
- **train_accuracy/val_accuracy**: Accuracy (%) on training and validation sets
- **samples_processed**: Number of training samples processed in the epoch
- **epoch_time**: Time taken for this epoch (seconds)
- **total_time**: Total training time so far (seconds)
- **learning_rate**: Current learning rate
- **cpu_memory_mb**: CPU memory usage in MB
- **gpu_memory_mb**: Current GPU memory usage in MB
- **gpu_peak_memory_mb**: Peak GPU memory usage in MB
- **batch_size**: Training batch size
- **device**: Training device (cuda:0, cpu, etc.)
- **Hardware info**: CPU model, GPU model, memory size, etc.

## Supported Datasets

The module works with multiple dataset types:
- **EMNIST**: Extended MNIST dataset with letters and digits
- **MNIST**: Classic handwritten digit dataset
- **Custom**: User-provided image datasets

See the [Datasets Module](../datasets/README.md) for more information on dataset management.
