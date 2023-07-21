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
        '''
        print("heres the result:", res)
        print("here's the expected label", label_vector)
        print("here's the computed loss ")
        '''
        loss = self.loss_func(label_vector, output)
        error = self.loss_func_prime(label_vector, output)
        '''
        print("here's the loss after one pass:", loss)
        print("here's the loss error after one passs:", error)
        '''
        for i, layer in enumerate(reversed(self.layers)):
            '''
            print("current layer:",  layer)
            print("Back propagation iteration:", i)
            '''
            error = layer.backpropagation(error)

        '''
        print("current error: ", error)
        print("prediction/ res", res)
        '''

    def evaluate(self, input_vector: np.ndarray, label_vector):
        '''
        '''
        res = self.forward(input_vector)
        print("error on current example: ", self.loss_func(res, label_vector))
        print("res: ", res)
        print("label_vector: ", label_vector)
