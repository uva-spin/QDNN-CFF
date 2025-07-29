import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

fns = F1F2()
calc = F_calc()

torch.manual_seed(42)

# Hyperparameters
n_qubits = 6
max_layers = 8
initial_layers = 2
entanglement_strength = 0.5

# PennyLane device
dev = qml.device("default.qubit", wires=n_qubits)

# Create entanglement ranges list per layer
def create_entanglement_ranges(n_layers, strength):
    if strength < 1:
        return [1] * n_layers
    else:
        return [2] * n_layers

entanglement_ranges = create_entanglement_ranges(max_layers, entanglement_strength)

# Quantum circuit accepting variable depth
def quantum_circuit(inputs, weights, depth):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    
    qml.templates.StronglyEntanglingLayers(
        weights[:depth],
        wires=range(n_qubits),
        ranges=entanglement_ranges[:depth]
    )
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Create QNode factory to rebuild QNode with new depth
def make_qnode(depth):
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def qnode(inputs, weights):
        return quantum_circuit(inputs, weights, depth)
    return qnode
    
class NormalizeToPi(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min_vals = x.min(dim=0, keepdim=True)[0]
        max_vals = x.max(dim=0, keepdim=True)[0]
        scale = (max_vals - min_vals).clamp(min=1e-8)
        normed = 2 * np.pi * (x - min_vals) / scale - np.pi
        return normed

# Full QuantumDNN model
class QuantumDNN(nn.Module):
    def __init__(self, initial_depth=initial_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.max_layers = max_layers
        self.depth = initial_depth
        
        self.normalize = NormalizeToPi()
        
        self.preprocess = nn.Sequential(
            nn.Linear(3, n_qubits, dtype=torch.float64)
        )
        
        # Scale weights as before
        layer_scales = 1 / np.sqrt(np.arange(1, max_layers + 1))
        self.weights = nn.Parameter(
            torch.randn(max_layers, n_qubits, 3, dtype=torch.float64) * torch.tensor(layer_scales, dtype=torch.float64).view(-1,1,1)
        )
        
        self.postprocess = nn.Sequential(
            nn.Linear(n_qubits, 64, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(64, 4, dtype=torch.float64)
        )
        
        self.qnode = make_qnode(self.depth)
    
    def increase_depth(self):
        if self.depth < self.max_layers:
            self.depth += 1
            self.qnode = make_qnode(self.depth)
    
    def forward(self, inputs):
        inputs = inputs.to(torch.float64)
        x = self.normalize(inputs)
        x = self.preprocess(x)
        q_out = self.qnode(x, self.weights)
        q_out = torch.stack(q_out, -1)
        return self.postprocess(q_out) * 10

def modelf(phi_x, QQ, x_b, t, k, ReH, ReE, ReHt, dvcs):
    F_1, F_2 = fns.f1_f21(t)
    kins = [phi_x, QQ, x_b, t, k, F_1, F_2]
    cffs = [ReH, ReE, ReHt, dvcs]
    return calc.fn_1(kins, cffs)

def custom_loss(predicted_cffs, true_F, x_b, QQ, t, k, phi_x):
    ReH, ReE, ReHt, dvcs = torch.split(predicted_cffs, 1, dim=1)
    predicted_F = modelf(phi_x, QQ, x_b, t, k, ReH, ReE, ReHt, dvcs)
    loss = torch.mean((predicted_F - true_F.view_as(predicted_F)) ** 2)
    return loss


# Training loop with early stopping
results = {
    'Set #': [], 'Replica #': [], 
    'True ReH': [], 'Pred ReH': [],
    'True ReE': [], 'Pred ReE': [],
    'True ReHt': [], 'Pred ReHt': [],
    'True dvcs': [], 'Pred dvcs': [],
}

for j in range(len(error_bins)):
    model = QuantumDNN()
    optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
    
    phi_x = torch.tensor(error_bins[j]['phi_x'].values, dtype=torch.float32)
    QQ = torch.tensor(error_bins[j].iloc[0]['QQ'], dtype=torch.float32)
    x_b = torch.tensor(error_bins[j].iloc[0]['x_b'], dtype=torch.float32)
    t = torch.tensor(error_bins[j].iloc[0]['t'], dtype=torch.float32)
    k = torch.tensor(error_bins[j].iloc[0]['k'], dtype=torch.float32)
    true_F = torch.tensor(error_bins[j]['F'].values, dtype=torch.float32)

    X = torch.stack([x_b.unsqueeze(0), QQ.unsqueeze(0), t.unsqueeze(0)], dim=1).float()

    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predicted_cffs = model(X)
        loss = custom_loss(predicted_cffs, true_F, x_b, QQ, t, k, phi_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 15 == 0:  # Increase depth every 15 epochs
            model.increase_depth()

    print("Training complete for bin:", i, j+1)
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    print(predicted_cffs)
    print("True parameters:", [round(error_bins[j].iloc[0]['ReH'], 3), 
        round(error_bins[j].iloc[0]['ReE'], 3), 
        round(error_bins[j].iloc[0]['ReHt'], 3), 
        round(error_bins[j].iloc[0]['dvcs'], 3)])

    true_params = [error_bins[j].iloc[0]['ReH'], error_bins[j].iloc[0]['ReE'],
        error_bins[j].iloc[0]['ReHt'], error_bins[j].iloc[0]['dvcs']]
    pred_params = [predicted_cffs[:, 0].item(), predicted_cffs[:, 1].item(),
        predicted_cffs[:, 2].item(), predicted_cffs[:, 3].item()]
    #errors = [(true_params[k] - pred_params[k]) / true_params[k] * 100 for k in range(4)]

    results['Set #'].append(i)
    results['Replica #'].append(j+1)
    results['True ReH'].append(true_params[0])
    results['True ReE'].append(true_params[1])
    results['True ReHt'].append(true_params[2])
    results['True dvcs'].append(true_params[3])
    results['Pred ReH'].append(pred_params[0])
    results['Pred ReE'].append(pred_params[1])
    results['Pred ReHt'].append(pred_params[2])
    results['Pred dvcs'].append(pred_params[3])
