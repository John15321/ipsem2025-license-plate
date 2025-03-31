"""
Simple hybrid quantum-classical neural network for EMNIST dataset using 6 qubits.
This is a minimal implementation for easy analysis of the hybrid model architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Simple dataset preparation
print("Loading MNIST dataset (subset)...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),  # Resized to 64x64 to match CNN architecture
])

# Load a small subset of MNIST for quick analysis
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Use only first 100 samples for quick analysis
train_subset = torch.utils.data.Subset(train_dataset, range(100))
test_subset = torch.utils.data.Subset(test_dataset, range(20))

train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=5, shuffle=False)

print(f"Dataset loaded: {len(train_subset)} training samples, {len(test_subset)} test samples")

# 2. Simple 6-qubit quantum neural network
def create_quantum_nn():
    """Create a simple 6-qubit quantum neural network."""
    n_qubits = 6
    
    # Feature map for encoding classical data into quantum states
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    
    # Simple variational circuit with linear entanglement pattern
    ansatz = RealAmplitudes(num_qubits=n_qubits, entanglement="linear", reps=1)
    
    # Combine into a single circuit
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    # Create quantum neural network - explicitly enable gradient calculation for hybrid model
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,  # Essential for backpropagation in hybrid models
        estimator=StatevectorEstimator()
    )
    
    print(f"Quantum Neural Network created with {n_qubits} qubits")
    print(f"Input parameters: {len(feature_map.parameters)}")
    print(f"Weight parameters: {len(ansatz.parameters)}")
    
    return qnn

# 3. Hybrid model - combines classical NN with quantum NN
class SimpleHybridModel(nn.Module):
    def __init__(self, n_classes=10, n_qubits=6, input_channels=1):
        super().__init__()
        
        # 1. Classical CNN feature extraction - using the provided architecture
        self.classical_part = nn.Sequential(
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
        
        # 2. Quantum part
        qnn = create_quantum_nn()
        self.quantum_part = TorchConnector(qnn)
        
        # 3. Classical output layer
        self.post_quantum = nn.Linear(1, n_classes)  # Map quantum output to classes
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Classical pre-processing with CNN
        classical_output = self.classical_part(x)
        
        # Process each sample through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            quantum_output = self.quantum_part(classical_output[i])
            quantum_outputs.append(quantum_output)
            
        # Combine outputs and reshape
        q_out = torch.stack(quantum_outputs).reshape(batch_size, 1)
        
        # Classical post-processing
        return self.post_quantum(q_out)

# 4. Training function
def simple_training_loop(model, train_loader, epochs=2):
    """A minimal training loop for demonstration."""
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/5:.4f}')
                running_loss = 0.0
    
    print('Training finished')
    return model

# 5. Simple evaluation function
def simple_evaluation(model, test_loader):
    """A minimal evaluation loop."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    
    return accuracy

# 6. Main execution
if __name__ == "__main__":
    print("Creating hybrid quantum-classical model...")
    model = SimpleHybridModel(n_classes=10, n_qubits=6, input_channels=1)  # MNIST has 10 classes
    
    print("\nModel architecture:")
    print("1. Classical pre-processing (CNN):")
    print("   Input image (64x64) → 3x[Conv2D→ReLU→MaxPool2D] → Flatten → Linear(8192→256) → ReLU → Dropout(0.5) → Linear(256→6) → Tanh")
    print("2. Quantum processing:")
    print("   6 features → 6-qubit quantum circuit → 1 measurement")
    print("3. Classical post-processing:")
    print("   1 quantum output → Linear(1→10) → 10 classes\n")
    
    print("Training model (minimal training to demonstrate functionality)...")
    model = simple_training_loop(model, train_loader, epochs=1)
    
    print("Evaluating model...")
    accuracy = simple_evaluation(model, test_loader)
    
    print("Saving model...")
    torch.save(model.state_dict(), "simple_hybrid_model.pt")
    print("Model saved to simple_hybrid_model.pt")