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
    

#Now let's construct our neural network
nn = Network(activation_functions.cross_entropy, activation_functions.cross_entropy_prime)
nn.addLayer(DenseLayer(2, 2, 1, activation_functions.tanh, activation_functions.tanh_prime))
nn.addLayer(General_dense_layer(2, 2, 2, activation_functions.softmax, activation_functions.soft_max_prime))

epochs = 10
#Now we will go through and train
for epoch in range(epochs):
    print("epoch ", epoch)
    for i in range(len(X_xor)):
        training_vector = X_xor[i]
        label_vector = one_hot_encoder(y_xor[i])
        nn.train(training_vector, label_vector)
        if epoch % 1== 0:
            nn.evaluate(training_vector, label_vector)
    if epoch % 100 == 0:
        print("\n\n\n")

