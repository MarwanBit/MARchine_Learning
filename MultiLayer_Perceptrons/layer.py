import numpy as np


class BaseLayer():
    def __init__(self, input: np.ndarray):
        '''
        Base class for which we will build up multiple layers classes
        '''
        self.input = None
        self.output = None


    def forward(self, input):
        '''
        '''
        raise NotImplementedError


    def backpropagation(self, output_error, learning_rate):
        '''
        '''
        raise NotImplementedError


class DenseLayer(BaseLayer):
    def __init__(self, input_size: int, output_size: int, layer_num: int,  learning_rate: float = 0.01, input: np.ndarray = None):
        '''
        We initially start by creating a Weights matrix and bias matrix


        Note that we calculate the forward propagation by the formula 

        Y^(l+1)(n) = W^(l+1)(n)Y^(l)(n) + b^(l+1)(n)

        Where W is the weights matrix of the layer, Y^(l)(n) is the output of the previous layer,
        b^(l+1)(n) is the bias vector of the current layer, the dimensions of W is output_size * input_size, 
        the dimensions of Y^(l)(n) is going to be input size * 1, and the dimensions of b^(l+1)(n) is going to be output_size
        by 1. 

        '''
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)
        self.lr = learning_rate


    def forward(self, input: np.ndarray):
        '''
        Here we perform forward propagation using the rule that Y^(l+1)(n) = W^(l+1)(n)Y^(l)(n) + b^(l+1)(n)
        '''
        self.input = input
        self.output = np.matmul(self.weights,self.input) + self.bias 
        return self.output 
    

    def backpropagation(self, output_error: np.ndarray, learning_rate = None):
        '''
        '''
        #If no learning rate is passed than we use the one provided during initalization
        if learning_rate:
            self.lr = learning_rate
        weight_error = np.dot(self.input.T, output_error)
        bias_error = output_error
        input_error = np.dot(output_error, self.weights.T)
        #updates the weights
        self.weights -= self.lr*weight_error 
        self.bias -= self.lr* bias_error
        #Pass dE/dX as the de/DY for the next layer
        return input_error
    

class ActivationLayer(BaseLayer):
    def __init__(self, activation, activation_prime):
        '''
        '''
        self.activation = activation
        self.activation_prime = activation_prime


    def forward(self, input: np.ndarray):
        '''
        '''
        self.input = input 
        self.output = self.activation(input)
        return self.output
    

    def backpropagation(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)
