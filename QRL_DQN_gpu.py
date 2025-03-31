
#!/usr/bin/env python
# coding: utf-8
import gymnasium as gym
import math
import random
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
from torch import long as torch_long
from torch import float as torch_float
from torch import float32 as torch_float32
from torch import bool as torch_bool
from torch import tensor as torch_tensor
from torch import cat as torch_cat
from torch import zeros as torch_zeros
from torch.nn.utils import clip_grad_value_ as torch_clip_grad_value
from torch import no_grad
from torch import Tensor
import torch.nn as nn
from torch.optim import LBFGS, Adam


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print(torch.cuda.is_available(),"if True, then GPU is available-important!")

from qiskit  import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import BackendSampler
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator as Estimator, Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.connectors.torch_connector import Module
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 420

qi = BackendSampler(Aer.get_backend('statevector_simulator'))
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

problem_chosed = "CartPole-v1"
env = gym.make(problem_chosed)
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNN_DQN(Module):
    def __init__(self, QNN=None):
        super().__init__()
        if QNN is None:
            feature_map = ZZFeatureMap(number_of_inputs)
            ansatz = RealAmplitudes(number_of_inputs, entanglement='full', reps=2)
            qc = QuantumCircuit(number_of_inputs)
            qc.append(feature_map, range(number_of_inputs))
            qc.append(ansatz, range(number_of_inputs))

            qnn1 = SamplerQNN(
                circuit=qc,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                sampler=Sampler(),
                sparse=False,
            )

            initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
            self.qnn = TorchConnector(qnn1, initial_weights=initial_weights)
        else:
            self.qnn = QNN
        self.qnn.to(device)
        self.output_layer = nn.Linear(int(math.pow(2, number_of_inputs)), number_of_outputs).to(device)

    def forward(self, x):
        x = x.to(device)
        x_norm = (x * Tensor([0.1041666, 0.1041666, 1.1961722, 1.1961722]).to(device)) + Tensor([0.5, 0.5, 0.5, 0.5]).to(device)
        out = nn.functional.leaky_relu(self.qnn(x_norm))
        return nn.functional.leaky_relu(self.output_layer(out))


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1500
TAU = 0.01
LR = 5e-3

policyQNN = QNN_DQN().to(device)
targetQNN = QNN_DQN().to(device)
targetQNN.load_state_dict(policyQNN.state_dict())

optimizer = Adam(policyQNN.parameters(), lr=LR)
f_loss = nn.MSELoss(reduction="mean")
memory = ReplayMemory(10000)

steps_done = 0
episode_durations = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with no_grad():
            return policyQNN(state.to(device)).max(1).indices.view(1, 1)
    else:
        return torch_tensor([[env.action_space.sample()]], dtype=torch_long).to(device)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch_tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch_bool).to(device)
    non_final_next_states = torch_cat([s.to(device) for s in batch.next_state if s is not None])
    state_batch = torch_cat([s.to(device) for s in batch.state])
    action_batch = torch_cat([a.to(device) for a in batch.action])
    reward_batch = torch_cat([r.to(device) for r in batch.reward])

    state_action_values = policyQNN(state_batch).gather(1, action_batch)

    next_state_values = torch_zeros(BATCH_SIZE, device=device)
    with no_grad():
        next_state_values[non_final_mask] = targetQNN(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss(reduction="mean")
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    if torch.isnan(loss):
        return

    optimizer.zero_grad()
    try:
        loss.backward()
    except Exception as e:
        print("Error during backward:", str(e))
        return

    torch.nn.utils.clip_grad_norm_(policyQNN.parameters(), 1)
    optimizer.step()

plt.figure(figsize=(8, 6))
from IPython import display

def plot_durations(show_result=False):
    durations_t = torch_tensor(episode_durations, dtype=torch_float)
    if not show_result:
        plt.clf()
        plt.title('Training...')
    else:
        plt.clf()
        plt.title('Result')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch_cat((torch_zeros(99), means))
        plt.plot(means.numpy())

    plt.grid(True)
    plt.tight_layout()
    display.clear_output(wait=True)
    display.display(plt.gcf())

num_episodes = 100
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch_tensor(state, dtype=torch_float32).unsqueeze(0).to(device)
    print(f"Episode {i_episode} started")
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch_tensor([reward], device=device)
        done = terminated or truncated

        next_state = None if terminated else torch_tensor(observation, dtype=torch_float32).unsqueeze(0).to(device)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        target_net_state_dict = targetQNN.state_dict()
        policy_net_state_dict = policyQNN.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        targetQNN.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if i_episode % 1 == 0:
                print(f"Episode {i_episode} finished after {t + 1} steps")
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.grid(True)
plt.tight_layout()
plt.ioff()
plt.show()
