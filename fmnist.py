# ============================================================
# FMNIST Classification with Spiking CNN (snnTorch)
# Author: Nesara Shree
# Trains a Leaky Integrate-and-Fire spiking CNN (hybrid) on 
# FMNIST using surrogate gradient learning and BPTT.
# ============================================================

# ------------------------
# IMPORTS
# ------------------------
import snntorch as snn
from snntorch import surrogate, backprop, functional as SF, utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools


# ------------------------
# DATA PREPARATION
# ------------------------
# - Download and normalize fmnist (1-channel 28x28 images)
# - Create small subsets for quick experimentation

batch_size = 128
data_path = 'data/ffmnist'  # You can rename this folder or just use 'data/fmnist'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))  # fmnist mean=0, std=1 normalization
])

fmnist_train = datasets.fmnist(data_path, train=True, download=True, transform=transform)
fmnist_test = datasets.fmnist(data_path, train=False, download=True, transform=transform)

indices10k = torch.arange(1000)  # Subset first 1000 for train
indices5k = torch.arange(500)    # Subset first 500 for test
fmnist_train_10k = Subset(fmnist_train, indices10k)
fmnist_test_5k = Subset(fmnist_train, indices5k)

print(f"Full training set size: {len(fmnist_train)}")
print(f"Full test set size:     {len(fmnist_test)}")
print(f"Subset training size:   {len(fmnist_train_10k)}")
print(f"Subset test size:       {len(fmnist_test_5k)}")

train_loader = DataLoader(fmnist_train_10k, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(fmnist_test_5k, batch_size=batch_size, shuffle=True, drop_last=True)


# ------------------------
# NETWORK DEFINITION
# ------------------------
# - CNN + LIF neurons with surrogate gradient (fast sigmoid)
# - Architecture:
#   Conv(1→12) + MaxPool + LIF →
#   Conv(12→64) + MaxPool + LIF →
#   Flatten → Linear(64*4*4 → 10) + LIF output layer

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5          # membrane decay constant
num_steps = 50      # number of time steps to simulate

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
    """
    Simulate the spiking CNN for num_steps time steps.

    Args:
        net (nn.Module): spiking CNN model
        num_steps (int): simulation time steps
        data (Tensor): batch of input images

    Returns:
        spk_rec (Tensor): spike outputs recorded over time
        mem_rec (Tensor): membrane potentials recorded over time
    """
    spk_rec = []
    mem_rec = []

    utils.reset(net)  # Reset hidden states for all LIF neurons

    for _ in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


# ------------------------
# LOSS AND METRICS
# ------------------------
loss_fn = SF.ce_rate_loss()

def batch_accuracy(loader, net, num_steps):
    """
    Calculate accuracy of the network over a data loader.

    Args:
        loader (DataLoader): data loader for evaluation
        net (nn.Module): trained spiking CNN
        num_steps (int): simulation steps per batch

    Returns:
        float: accuracy (0 to 1)
    """
    net.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            batch_acc = SF.accuracy_rate(spk_rec, targets)
            correct += batch_acc * spk_rec.size(1)
            total += spk_rec.size(1)

    return correct / total


# ------------------------
# INITIAL TEST BEFORE TRAINING
# ------------------------
data, targets = next(iter(train_loader))
data, targets = data.to(device), targets.to(device)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_val = loss_fn(spk_rec, targets)
acc_val = SF.accuracy_rate(spk_rec, targets)

print(f"Initial loss (untrained):    {loss_val.item():.3f}")
print(f"Initial accuracy (1 batch):  {acc_val * 100:.2f}%")

test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"Initial test accuracy:        {test_acc * 100:.2f}%")


# ------------------------
# TRAINING LOOP
# ------------------------
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 10
test_acc_hist = []

for epoch in range(num_epochs):
    avg_loss = backprop.BPTT(
        net, train_loader,
        optimizer=optimizer,
        criterion=loss_fn,
        num_steps=num_steps,
        time_var=False,
        device=device
    )

    print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

    test_acc = batch_accuracy(test_loader, net, num_steps)
    test_acc_hist.append(test_acc)

    print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")
