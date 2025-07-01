# IMPORTS
'''
loads libraries needed for: 
1. SNN components (snntorch) Standard PyTorch operations (torch, torchvision)
2. Data loading and preprocessing
3. Plotting (matplotlib)
'''
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

# DATA PREPARATION
'''
Downloads and normalizes CIFAR-10 (3-channel 32x32 images, 10 classes). Creates two small subsets:
1. cifar_train_10k: first 1,000 samples
2. cifar_test_5k: first 500 samples
'''
batch_size = 128
data_path='data/cifar'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose(    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),])

cifar_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

indices10k = torch.arange(1000)
indices5k = torch.arange(500)
cifar_train_10k = Subset(cifar_train, indices10k)
cifar_test_5k = Subset(cifar_train, indices5k)

print(f"cifar_train is: {len(cifar_train):.2f}%")
print(f"cifar_test is: {len(cifar_test):.2f}%")
print(f"cifar_train is: {len(cifar_train_10k):.2f}%")
print(f"cifar_test is: {len(cifar_test_5k):.2f}%")

'''
Creates PyTorch DataLoaders for batching.
'''
#train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, drop_last=True)
#test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(cifar_train_10k, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(cifar_test_5k, batch_size=batch_size, shuffle=True, drop_last=True)

# DEFINE NETWORK
'''
Sets neuron type to Leaky (i.e., LIF neurons) with a fast surrogate gradient (fast_sigmoid) for spike-based backpropagation.
Defines beta (membrane decay) and number of time steps (num_steps=50) for temporal simulation.
'''
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

'''
A CNN + SNN hybrid model: Uses snn.Leaky neurons at each layer to model spiking behavior.
Conv2d(3 → 12) → LIF → MaxPool
Conv2d(12 → 64) → LIF → MaxPool
Flatten → Linear → LIF
'''
class LIF(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        beta = 0.95
        spike_grad = surrogate.atan()
        self.batch_size = batch_size

        # Initialize layers
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 5 * 5, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

# FORWARD PASS
'''
The forward() method applies convolution, pooling, LIF neurons, and dense layers over a single time step. It returns spike outputs and membrane potentials.
1. Runs the model over num_steps=50 time steps.
2. Stores spike and membrane output at each step.
3. Resets network state before each new sequence using utils.reset(net).
'''
    def forward(self, x):
        # Initialize hidden states and outputs at t=0
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

# MODEL
net = LIF(batch_size=batch_size).to(device=device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

for step in range(num_steps):
    spk_out, mem_out = net(data)

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

# LOSS AND ACCURACY
'''
Loss function: SF.ce_rate_loss() = cross-entropy on average spike rates.
Accuracy: Calculated as whether the highest average spike rate matches the true label (SF.accuracy_rate()).
batch_accuracy(): evaluates network accuracy across an entire dataset. Uses torch.no_grad() to avoid tracking gradients during testing.
'''
# already imported snntorch.functional as SF 
loss_fn = SF.ce_rate_loss()

loss_val = loss_fn(spk_rec, targets)
print(f"The loss from an untrained network is {loss_val.item():.3f}")

acc = SF.accuracy_rate(spk_rec, targets)
print(f"The accuracy of a single batch using an untrained network is {acc*100:.3f}%")

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

test_acc = batch_accuracy(test_loader, net, num_steps)

print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 100
test_acc_hist = []

# TRAINING LOOP
'''
Uses Backpropagation Through Time (BPTT) from snntorch.backprop to train the network on spike-based outputs.
Optimizer: Adam, learning rate 1e-2. 100 epochs.
Tracks and prints average training loss + accuracy on test set each epoch
'''
for epoch in range(num_epochs):

    avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn, 
                            num_steps=num_steps, time_var=False, device=device)
    
    print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

    # TEST SET ACCURACY
    test_acc = batch_accuracy(test_loader, net, num_steps)
    test_acc_hist.append(test_acc)

    print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")
