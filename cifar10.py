# ============================================================
# CIFAR-10 Classification with Spiking CNN (snnTorch)
# Author: Nesara Shree
# Trains a Leaky Integrate-and-Fire spiking CNN (hybrid) on 
# CIFAR-10 using surrogate gradient learning and BPTT.
# ============================================================

# ------------------------
# IMPORTS
# ------------------------
# - snnTorch for spiking neuron layers, loss functions, and training
# - PyTorch for tensor ops, layers, data loading
# - torchvision for CIFAR-10 dataset
# - matplotlib for plotting (optional)

import snntorch as snn
from snntorch import surrogate, backprop, functional as SF, utils, spikeplot as splt

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
# - Download and normalize CIFAR-10 (3-channel, 32x32 images, 10 classes)
# - Create small training/testing subsets for fast experimentation

batch_size = 128
data_path = 'data/cifar'
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

cifar_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

indices10k = torch.arange(1000)
indices5k = torch.arange(500)
cifar_train_10k = Subset(cifar_train, indices10k)
cifar_test_5k = Subset(cifar_train, indices5k)

print(f"Full training set size: {len(cifar_train)}")
print(f"Full test set size:    {len(cifar_test)}")
print(f"Subset training size:  {len(cifar_train_10k)}")
print(f"Subset test size:      {len(cifar_test_5k)}")

train_loader = DataLoader(cifar_train_10k, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(cifar_test_5k, batch_size=batch_size, shuffle=True, drop_last=True)


# ------------------------
# NETWORK DEFINITION
# ------------------------
# - Uses LIF neurons with surrogate gradient (atan)
# - CNN + SNN hybrid using Leaky Integrate-and-Fire (LIF) neurons.
# - CNN backbone: Conv2D → MaxPool → LIF × 2 → FC → LIF

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5          # membrane decay
num_steps = 50      # simulation time steps

class LIF(nn.Module):
    """
    Architecture:
        Conv(3→12) + MaxPool + LIF →
        Conv(12→64) + MaxPool + LIF →
        FC(64*5*5 → 10) + LIF
    """

    def __init__(self, batch_size):
        super().__init__()
        beta = 0.95
        spike_grad = surrogate.atan()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 5 * 5, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(self.batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3


# ------------------------
# FORWARD PASS OVER TIME
# ------------------------
# - Simulates network over `num_steps`
# - Stores spike + membrane activity at each step

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # Reset internal states

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


# ------------------------
# LOSS + METRICS
# ------------------------
# - Loss: cross-entropy on average spike rate
# - Accuracy: percent correct from spiking output

loss_fn = SF.ce_rate_loss()

def batch_accuracy(loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


# ------------------------
# INITIAL EVALUATION (Untrained)
# ------------------------
net = LIF(batch_size=batch_size).to(device=device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_val = loss_fn(spk_rec, targets)
acc_val = SF.accuracy_rate(spk_rec, targets)

print(f"Initial loss (untrained):  {loss_val.item():.3f}")
print(f"Initial accuracy (1 batch): {acc_val * 100:.2f}%")

test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"Initial test accuracy:      {test_acc * 100:.2f}%")


# ------------------------
# TRAINING LOOP
# ------------------------
# - Trains using BPTT (backpropagation through time) from snntorch.backprop
# - Optimizer: Adam
# - Logs training loss and test accuracy per epoch

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 100
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
