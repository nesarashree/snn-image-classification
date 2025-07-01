# ============================================================
# MNIST Spiking CNN Training with snnTorch
# ============================================================

# ------------------------
# IMPORTS
# ------------------------
import snntorch as snn
from snntorch import surrogate, backprop, functional as SF, utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

# ------------------------
# HYPERPARAMETERS & DEVICE
# ------------------------
batch_size = 128
data_path = 'data/mnist'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------
# DATA PREPARATION
# ------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))  # MNIST mean=0, std=1 normalization
])

# Download datasets
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Use small subsets for quick experiments
indices_train = torch.arange(1000)
indices_test = torch.arange(500)
mnist_train_10k = Subset(mnist_train, indices_train)
mnist_test_5k = Subset(mnist_test, indices_test)

print(f"Full training size: {len(mnist_train)}")
print(f"Full test size: {len(mnist_test)}")
print(f"Subset training size: {len(mnist_train_10k)}")
print(f"Subset test size: {len(mnist_test_5k)}")

# DataLoaders
train_loader = DataLoader(mnist_train_10k, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test_5k, batch_size=batch_size, shuffle=True, drop_last=True)

# ------------------------
# NETWORK & SIMULATION PARAMETERS
# ------------------------
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

# Define network architecture with Leaky LIF neurons
net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Conv2d(12, 64, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)

# ------------------------
# FORWARD PASS FUNCTION
# ------------------------
def forward_pass(net, num_steps, data):
    """Simulate the network over multiple time steps."""
    spk_rec, mem_rec = [], []
    utils.reset(net)  # reset neuron states

    for _ in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

# ------------------------
# LOSS AND ACCURACY FUNCTIONS
# ------------------------
loss_fn = SF.ce_rate_loss()

def batch_accuracy(loader, net, num_steps):
    """Calculate accuracy over the entire data loader."""
    net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)
            acc_batch = SF.accuracy_rate(spk_rec, targets)
            correct += acc_batch * spk_rec.size(1)
            total += spk_rec.size(1)
    return correct / total

# ------------------------
# INITIAL EVALUATION
# ------------------------
data, targets = next(iter(train_loader))
data, targets = data.to(device), targets.to(device)

spk_rec, mem_rec = forward_pass(net, num_steps, data)
loss_val = loss_fn(spk_rec, targets)
acc_val = SF.accuracy_rate(spk_rec, targets)

print(f"Initial loss (untrained): {loss_val.item():.3f}")
print(f"Initial accuracy (1 batch): {acc_val*100:.2f}%")

test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"Initial test accuracy: {test_acc*100:.2f}%")

# ------------------------
# TRAINING LOOP
# ------------------------
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 10
test_acc_hist = []

for epoch in range(num_epochs):
    avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn, 
                             num_steps=num_steps, time_var=False, device=device)
    print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

    test_acc = batch_accuracy(test_loader, net, num_steps)
    test_acc_hist.append(test_acc)
    print(f"Epoch {epoch}, Test Acc: {test_acc*100:.2f}%\n")
