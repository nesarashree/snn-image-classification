## Spiking Neural Networks for Computer Vision (CIFAR-10, FMNIST, MNIST)
This is a comprehensive training script for a Spiking Neural Network (SNN) using the snntorch library on the CIFAR-10 dataset, specifically employing a Leaky Integrate-and-Fire (LIF) neuron model.

### INITIAL SETUP
Do this only once. When you log in again, go to your home directory and run the command `source sosenv/bin/activate`.
```
# SETUP ENVIRONMENT
mkdir /home/snn/ 
cd /home/snn/

# Create virtual environment and connect
python3 -m venv --copies snnenv
source sosenv/bin/activate
#after previous step, you should see (snnenv), prefixed to terminal
pip install --upgrade pip

# install PyTorch and snnTorch 
pip install torch >= 1.1.0
pip install numpy >= 1.17
pip install pandas
pip install matplotlib
pip install 
pip install snntorch
```
### TRAINING
```
# train MNIST
python mnist.py
# train FMNIST
python fmnist.py
# train CIFAR-10
python cifar10.py
```
