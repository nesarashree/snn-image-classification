## Step-by-step Instructions

### INITIAL SETUP

Do this only once. When you login again, go to home directory and run the command `source sosenv/bin/activate`.

```
#SETUP ENVIRONMENT
#home directory
mkdir /home/snn/
cd /home/snn/
#Create virtual environment and connect
python3 -m venv --copies snnenv
source sosenv/bin/activate
#after previous step, you should see (snnenv), prefixed to terminal
pip install --upgrade pip

#install PyTorch
pip install torch >= 1.1.0
pip install numpy >= 1.17
pip install pandas
pip install matplotlib
pip install 
pip install snntorch

#you are all set
```

### TRAINING

## train mnist
```
python mnist.py
```

## train fmnist
```
python fmnist.py
```

## train fmnist
```
python cifar10.py
```
