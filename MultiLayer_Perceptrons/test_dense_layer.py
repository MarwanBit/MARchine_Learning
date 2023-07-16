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

def forward_propagation_test_2():
    '''
    Check if forward propagation will work on a network that is a little bit more complicated
    '''
    XOR_dense_1 = DenseLayer(2, 2, 1)
    Activation_layer = ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime)
    XOR_dense_2 = DenseLayer(2, 1, 2)
    Activation_layer_2 = ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime)
    v_1 = XOR_dense_1.forward(X_xor[0])
    print(v_1)
    y_1 = Activation_layer.forward(v_1)
    print(y_1)
    v_2 = XOR_dense_2.forward(y_1)
    print(v_2)
    y_2 = Activation_layer_2.forward(v_2)
    print(y_2)
    assert y_2.shape == (1,1)

def forward_propagation_test_3():
    '''
    This checks that forward propagation will work on a neural network which we construct
    '''
    nn = Network(activation_functions.mse, activation_functions.mse_prime)
    XOR_dense_1 = DenseLayer(2, 2, 1)
    Activation_layer = ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime)
    XOR_dense_2 = DenseLayer(2, 1, 2)
    Activation_layer_2 = ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime)
    nn.addLayer(XOR_dense_1)
    nn.addLayer(Activation_layer)
    nn.addLayer(XOR_dense_2)
    nn.addLayer(Activation_layer_2)
    print(nn.forward(X_xor[0]))
    


def neural_network_test_1():
    '''
    '''
    nn = Network(activation_functions.mse, activation_functions.mse_prime)
    nn.addLayer(DenseLayer(2, 2, 1))
    nn.addLayer(ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime))
    nn.addLayer(DenseLayer(2, 1, 2))
    nn.addLayer(ActivationLayer(activation_functions.tanh, activation_functions.tanh_prime))
    for i in range(10000):
        for i in range(len(X_xor)):
            training_vector, label_vector = X_xor[i], y_xor[i]
            nn.train(training_vector, label_vector)
            nn.evaluate(training_vector, label_vector)





if __name__ == "__main__":
    print("Completing forward propogation test 1 !")
    forward_propagation_test_1()
    print("Completing forward propagation test 2!")
    forward_propagation_test_2()
    print("Completing forward propagation test 3")
    forward_propagation_test_3()
    print("testing neural network test 1!")
    neural_network_test_1()