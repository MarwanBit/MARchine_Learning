from layer import DenseLayer
import numpy as np

class General_dense_layer(DenseLayer):
    def __init__(self, input_size: int, output_size: int, layer_num: int, activation_function, activation_function_prime,
                   learning_rate: float = 0.01, input: np.ndarray = None):
        super().__init__(input_size, output_size, layer_num, activation_function, activation_function_prime, learning_rate, input)

    def forward(self, input: np.ndarray):
        return super().forward(input)

    def backpropagation(self, output_error: np.ndarray, learning_rate=0.01):
        '''
        Generalized backpropagation formulas for non-element wise activation functions 
        '''
        weight_error = np.matmul(output_error.T, self.activation_function_prime(self.local_field))
        weight_error = np.matmul(weight_error.T, self.input.T)
        bias_error = np.matmul(output_error.T, self.activation_function_prime(self.local_field))
        input_error = np.matmul(output_error.T, np.matmul(self.activation_function_prime(self.local_field), self.weights))
        self.weights -= weight_error
        self.bias -= bias_error.T
        print("weights error ", weight_error)
        print(weight_error.shape)
        print("input_error ", input_error)
        print(input_error.shape)
        print("bias error ", bias_error)
        print(bias_error.shape)
        return input_error