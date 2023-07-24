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
    def __init__(self, input_size: int, output_size: int, layer_num: int, activation_function, activation_function_prime,
                   learning_rate: float = 0.01, input: np.ndarray = None):
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
        self.local_field = None
        self.activation_function = activation_function 
        self.activation_function_prime = activation_function_prime


    def forward(self, input: np.ndarray):
        '''
        Here we perform forward propagation using the rule that Y^(l+1)(n) = W^(l+1)(n)Y^(l)(n) + b^(l+1)(n)
        '''
        self.input = input
        self.local_field = np.matmul(self.weights, self.input) + self.bias
        self.output = self.activation_function(self.local_field) 
        return self.output 
    

    def backpropagation(self, output_error: np.ndarray, learning_rate = 0.01):
        '''
        '''
        #If no learning rate is passed than we use the one provided during initalization
        #check this stuff 
        print("output_error.T shape, ", output_error.T.shape)
        print("self.activation_function_prime shape:, ", self.activation_function_prime(self.local_field).shape)
        weight_error = np.multiply(output_error.T, self.activation_function_prime(self.local_field))
        print("weight error shape: ", weight_error.shape)
        print("self.input.T shape, ", self.input.T.shape)
        weight_error = np.matmul(weight_error, self.input.T)
        bias_error = np.multiply(output_error.T, self.activation_function_prime(self.local_field))
        input_error = np.multiply(output_error, self.activation_function_prime(self.local_field).T)
        input_error = np.matmul(input_error, self.weights)
        #updates the weights
        self.weights -= self.lr*weight_error
        self.bias -= self.lr* bias_error
        #Pass dE/dX as the de/DY for the next layer
        return input_error
    


