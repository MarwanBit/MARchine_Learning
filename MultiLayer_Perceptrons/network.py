import layer as Layer 

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

    def train(self, input_vector, label_vector):
        '''
        '''
        output = input
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
