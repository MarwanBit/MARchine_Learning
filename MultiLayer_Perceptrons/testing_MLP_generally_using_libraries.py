import os 
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np 
from network import Network
import activation_functions
from layer import DenseLayer, BaseLayer
from general_dense_layer import General_dense_layer

#First let's construct the XOR Dataset 
#This is for the XOR Problem which we will use for experimentation
X_xor = np.array([
    np.array([[-1.0], 
              [-1.0]]),
    np.array([[-1.0], 
              [+1.0]]),
    np.array([[+1.0], 
              [-1.0]]),
    np.array([[+1.0], 
              [+1.0]]),
])
y_xor = np.array([
    np.array([[-1.0]]), 
    np.array([[1.0]]), 
    np.array([[1.0]]), 
    np.array([[-1.0]]),
    ])

#We want to construct a one hot encoder for the y_xor vectors
def one_hot_encoder(label_vector):
    if label_vector[0] == -1:
        return np.array([[1], [0]])
    elif label_vector[0] == 1:
        return np.array([[0], [1]])

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2*1, 2),
            nn.ReLU(),
            nn.Linear(2*1, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
network = NeuralNetwork()
for i in range(len(X_xor)):
    training_vector = X_xor[i]
    print(network.forward(training_vector))