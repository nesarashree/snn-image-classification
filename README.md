## Spiking Neural Networks for Image Classification
This repository contains comprehensive training scripts for Spiking Neural Networks (SNNs) using the [snnTorch](https://github.com/jessemelpolio/snntorch) library. The models utilize Leaky Integrate-and-Fire (LIF) neurons and are trained on classic computer vision datasets, including the publicly available [CIFAR-10](https://www.kaggle.com/c/cifar-10) and Fashion-MNIST ([FMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)).

### Initial Setup
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
### Training
Use the following commands to run training scripts for each dataset:
```
python fmnist.py      # Train on Fashion-MNIST
python cifar10.py     # Train on CIFAR-10
```

### Notes
* Make sure your working directory contains the respective .py scripts for training.
* Adjust batch sizes, epochs, and other hyperparameters in each script as needed.
* GPU support requires CUDA-enabled PyTorch installed.
