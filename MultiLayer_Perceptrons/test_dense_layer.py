'''
This file will be used to test the outputs of the dense layer using the example XOR dataset which 
we will declare below, but first let's import some functionality  
'''

import numpy as np 
from layer import DenseLayer, ActivationLayer, BaseLayer
from network import Network
import activation_functions

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


def forward_propagation_test_1():
    '''
    Let's us check to see that the forward propagation step is working correctly 
    '''
    XOR_dense = DenseLayer(2, 2, 1)
    #We have a dense layer with two neurons tested on an input layer previously with 2 inputs x_{1}(n) and 
    #x_{2}(n)
    assert XOR_dense.forward(X_xor[0]).shape == (2,1)


def neural_network_test_1():
    '''
    '''
    nn = Network(activation_functions.mse, activation_functions.mse_prime)
    nn.addLayer(DenseLayer(2, 2, 1))
    nn.addLayer(ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime))
    nn.addLayer(DenseLayer(2, 1, 2))
    nn.addLayer(ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime))
    nn.train(X_xor[0], y_xor[0])





if __name__ == "__main__":
    forward_propagation_test_1()
    neural_network_test_1()