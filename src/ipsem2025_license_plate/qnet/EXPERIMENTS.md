# Quantum Neural Network Experiment Manager

This module provides a systematic way to investigate how different quantum circuit configurations affect model performance. It allows for running multiple experiments with various combinations of qubits, ansatz depths, and feature map depths.

## Usage

The experiment manager provides two main commands:

### Run Experiments

Run a series of experiments with different quantum circuit configurations:

```bash
ipsem2025-experiments run \
    --qubits 4,6,8 \
    --ansatz 1-5 \
    --feature-map 1-3 \
    --epochs 10 \
    --batch-size 128 \
    --output-dir experiments \
    --name "depth_investigation" \
    --dataset-type emnist \
    --dataset-path data \
    --learning-rate 0.001 \
    --use-gpu \
    --preload-data \
    --num-workers 8
```

### Analyze Results

Analyze results from previous experiment runs:

```bash
ipsem2025-experiments analyze \
    --dir experiments/20250410_120000_depth_investigation \
    --output report.md
```

## Command Parameters

### Run Command

- `--qubits`, `-q`: Comma-separated list or range of qubits to test (e.g., '4,6,8' or '4-8')
- `--ansatz`, `-a`: Comma-separated list or range of ansatz repetitions (e.g., '1,2,3' or '1-5')
- `--feature-map`, `-f`: Comma-separated list or range of feature map repetitions (e.g., '1,2' or '1-3')
- `--epochs`, `-e`: Number of epochs for each experiment (default: 10)
- `--batch-size`, `-b`: Batch size for training (default: 128)
- `--output-dir`, `-o`: Base directory for experiment outputs (default: "experiments")
- `--name`, `-n`: Name for this experiment suite
- `--dataset-type`, `-d`: Dataset to use (default: "emnist")
- `--dataset-path`, `-p`: Path to dataset (default: "data")
- `--learning-rate`, `-l`: Learning rate (default: 0.001)
- `--use-gpu/--no-gpu`: Use GPU for quantum simulation (default: True)
- `--preload-data/--no-preload`: Preload dataset into memory (default: True)
- `--skip-existing`: Skip configurations that have already been run (default: False)
- `--num-workers`: Number of dataloader workers (default: auto)
- `--verbose`, `-v`: Enable verbose output (default: False)

### Analyze Command

- `--dir`, `-d`: Directory containing experiment results
- `--output`, `-o`: Output file for the analysis report

## Output Directory Structure

For each experiment run, a timestamped directory is created with the following structure:

