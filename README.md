# Uncertainty_Mnist
Uncertainty estimation on Mnist dataset

This is a PyTorch implementation of Dropout Uncertainty on Mnist. The experiment setting is based on [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) at 5.2 Model Uncertainty in Classification Tasks.

## Installation 

1. Install Pytorch
[Pytorch](http://pytorch.org/)
```
conda install pytorch torchvision -c pytorch
```
2. Clone this repository 
```
git clone https://github.com/andyhahaha/Uncertainty_Mnist
```

## Usage 

1. Train Lenet standard and Lenet dropout
```
python main.py --mode 0
```

2. Test Lenet standard and Lenet dropout
```
python main.py --mode 1
```

3. Test the Lenet dropout on rotated Mnist image 
```
python main.py --mode 2
```

## Result


