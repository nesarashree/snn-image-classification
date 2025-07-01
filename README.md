## Spiking Neural Networks for Computer Vision (CIFAR-10, FMNIST)
This repository contains comprehensive training scripts for Spiking Neural Networks (SNNs) using the [snnTorch](https://github.com/jessemelpolio/snntorch) library. The models utilize Leaky Integrate-and-Fire (LIF) neurons and are trained on classic computer vision datasets including CIFAR-10 and Fashion-MNIST (FMNIST).

### INITIAL SETUP
Run this setup **once** to create and activate a virtual environment with all dependencies.  
When you return to your system later, just run `source snnenv/bin/activate` from your home directory to activate the environment.
```
# Create directory for SNN projects and navigate into it
mkdir -p ~/snn && cd ~/snn

# Create virtual environment (copies dependencies for isolation)
python3 -m venv --copies snnenv

# Activate the virtual environment
source snnenv/bin/activate
# You should see (snnenv) prefix in your terminal prompt

# Upgrade pip inside the environment
pip install --upgrade pip

# Install required packages
pip install torch>=1.1.0
pip install numpy>=1.17
pip install pandas
pip install matplotlib
pip install snntorch
```
### TRAINING
Use the following commands to run training scripts for each dataset:
```
# train FMNIST
python fmnist.py

# train CIFAR-10
python cifar10.py
```

### NOTES
* Make sure your working directory contains the respective .py scripts for training.
* Adjust batch sizes, epochs, and other hyperparameters in each script as needed.
* GPU support requires CUDA-enabled PyTorch installed.
