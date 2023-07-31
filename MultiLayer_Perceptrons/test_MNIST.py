import numpy as np
from layer import DenseLayer, BaseLayer
from network import Network 
import activation_functions
from keras.datasets import mnist
import random
from general_dense_layer import General_dense_layer
import time


#Load the MNIST dataset 
(train_X, train_y), (test_X, test_y) = mnist.load_data()

def one_hot_encoder(label: int) -> np.ndarray:
    label_vector = np.zeros((10, 1))
    label_vector[label] = 1
    return label_vector

def reshape_X(training_vector: np.ndarray) -> np.ndarray:
    return training_vector.flatten().reshape(784, 1)


def MNIST_load():
    '''
    '''
    print("Train_X Shape: ", train_X.shape)
    print("Train_Y Shape: ", train_y.shape)
    print("test_X Shape: ", test_X.shape)
    print("test_y Shape: ", test_y.shape)

    #work on exploring the dataset!
    print("X_0 shape: ", train_X[0].shape)
    print("y_0 shape: ", train_y[0].shape)
    print("X_0 processed shape: ", train_X[0].flatten().reshape(784, 1).shape)
    print("one hot encoder : ", one_hot_encoder(train_y[0]))
    print("one hot encode shape : ", one_hot_encoder(train_y[0]).shape)


def MNIST_forward_test():
    '''
    '''
    
    #Create the Neural network
    nn = Network(activation_functions.cross_entropy, activation_functions.cross_entropy_prime)
    nn.addLayer(DenseLayer(784, 392, 1, activation_functions.tanh, activation_functions.tanh_prime))
    nn.addLayer(DenseLayer(392, 196, 2, activation_functions.tanh, activation_functions.tanh_prime))
    nn.addLayer(General_dense_layer(196, 10, 2, activation_functions.softmax, activation_functions.soft_max_prime))

    #training example
    training_vector = reshape_X(train_X[0])
    y_vector = one_hot_encoder(train_y[0])
    res = nn.forward(training_vector)
    print(res)
    print(res[0])
    assert res.shape == (10, 1)


def MNIST_training_test(epochs: int):
   #Create the Neural network
    nn = Network(activation_functions.cross_entropy, activation_functions.cross_entropy_prime)
    nn.addLayer(DenseLayer(784, 392, 1, activation_functions.tanh, activation_functions.tanh_prime))
    nn.addLayer(DenseLayer(392, 196, 2, activation_functions.tanh, activation_functions.tanh_prime))
    nn.addLayer(General_dense_layer(196, 10, 2, activation_functions.softmax, activation_functions.soft_max_prime))


    #Shuffling the dataset and training across it for 1 epoch
    for i in range(epochs):
        print("current epoch: ", i)
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        #This implements one_line_learning!
        for progress, index in enumerate(indices):
            training_vector = train_X[index]
            label_vector = train_y[index]
            #Process the data
            training_vector = reshape_X(training_vector)
            label_vector = one_hot_encoder(label_vector)
            if progress % 2000 == 0:
                print("training on example: ", progress, "out of ", len(train_X))
            nn.train(training_vector, label_vector)
    print("\n\n\n\n\n\n")
    print("finished training!")

    print("Beggining testing!")
    #Now let's test our neural network on some examples
    #Change this to len(train_x) if you want to do it on the whole set
    for i in range(2000):
        training_vector = test_X[index]
        label_vector = test_y[index]
        #Process the data
        training_vector = reshape_X(training_vector)
        label_vector = one_hot_encoder(label_vector)
        nn.evaluate(training_vector, label_vector)
        time.sleep(2)





if __name__ == "__main__":
    print("MNIST_Load testing!")
    MNIST_load()
    print("MNIST_forward_test testing!")
    MNIST_forward_test()
    print("MNIST_training_test testing!")
    MNIST_training_test(100)
