import qiskit
from qiskit_aer.primitives import SamplerV2

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

# Construct an ideal simulator with SamplerV2
sampler = SamplerV2()
job = sampler.run([circ], shots=128)
# Fixed typo in import
from qiskit import *
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator  # Make sure you import from qiskit_aer

# Perform an ideal simulation
result_ideal = job.result()
counts_ideal = result_ideal[0].data.meas.get_counts()
print("Counts(ideal):", counts_ideal)
# Additional torch-related imports
import torch
import torch.nn.functional as F
import torch.optim as optim
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator  # Make sure you import from qiskit_aer
from torch import cat, manual_seed, no_grad
from torch.nn import (
    Conv2d,
    Dropout2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    NLLLoss,
    ReLU,
    Sequential,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
