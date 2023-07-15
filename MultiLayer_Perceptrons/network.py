import layer as Layer 
import numpy as np

class Network():
    def __init__(self, loss_func, loss_func_prime):
        '''
        '''
        self.layers = []
        self.loss_func = loss_func
        self.loss_func_prime = loss_func_prime

    def addLayer(self, layer: Layer.BaseLayer):
        '''
        '''
        self.layers.append(layer)

    def calcLoss(self):
        '''
        '''
        raise NotImplementedError
    
    def forward(self, input_vector: np.ndarray):
        '''
        '''
        output = input_vector
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, input_vector: np.ndarray, label_vector):
        '''
        '''
        output = input_vector
        for layer in self.layers:
            output = layer.forward(output)
        res = output

        #Now we calculate the loss
        loss = self.loss_func(label_vector, output)
        error = self.loss_func_prime(label_vector, output)
        for layer in reversed(self.layers):
            error = layer.backpropagation(error)

        print("current error: ", error)
        print("prediction/ res", res)
