#This file includes the implementation of the Rosenblatt's Perceptron
import numpy as np

class RosenBlott_Perceptron():

    def __init__(self, input_length: int, weights_vector: np.ndarray, 
                 learning_rate: float = 1.0, training_length: int = 0):
        '''
        Here we initialize the RosenBlott Perceptron, here's a description of what is encoded
        in the Perceptron (Check Neural Networks and Learning Machines by Haykin for more information)

        weights_vector: a 1*n numpy vector which of the form [b, w_{1}, ...., w_{n-1}], we initialize it 
        to be the 0 vector, since this is the case in the convergence theorem if not already set

        bias: the bias (scalar) of the perceptron

        input_length: {n-1} (integer) which we determine

        learning_rate: (float) value which gives the time-step for gradient descent

        training_length: (int) the number of training_examples in the dataset
        '''

        self.input_length = input_length
        self.learning_rate = learning_rate
        self.weights_vector = weights_vector
        self.bias = self.weights_vector[0]
        self.training_index = 0
        self.training_length = training_length
        

    def process_training_vector(self, training_vector: np.ndarray) -> np.ndarray:
        '''
        Processes our training vector by pre-concatenating a 1, so that w*x outputs 
        a linear function (some preprocessing that must be done with each example trainign_vector)
        '''

        training_vector = np.concatenate([np.array([1]), training_vector])
        return training_vector

    
    def forward(self, training_example: tuple) -> float:
        '''
        Here we get the output of the form phi(w*x) where phi is the hard limiter, here's a description of 
        the inputs

        training_example: (a tuple containing a n-1*1 vector (the training example) with the annotation of the class 
        [an int which is 0 or 1])
        '''
        
        training_vector = training_example[0]
        #now we construct the training_vector
        training_vector = self.process_training_vector(training_vector)
        return np.dot(self.weights_vector, training_vector)


    def class_prediction(self, result: float) -> int:
        '''
        Given the result of (w^{t}x + b), spits out a class prediction of either 0 or 1
        '''

        if result >= 0:
            return 0
        elif result < 0:
            return 1 
        
    
    def misclassification(self, training_example: tuple) -> bool:
        '''
        A function given a training_example from the dataset determines if it was 
        correctly classified or not.
        '''

        training_vector = training_example[0]
        training_annotation = training_example[1]
        result = self.forward(training_example)
        prediction = self.class_prediction(result)

        if (prediction == 0) and (training_annotation == 1):
            print("predicted 0, but actually 1")
            return True 
        elif (prediction == 1) and (training_annotation == 0):
            print("predicted 1, but actually 0")
            return True
        else:
            print("classified correctly!")
            return False


    def print_weights(self) -> np.ndarray:
        '''
        prints out the weight vector to the console for visualization and additionally
        returns the weights vector for debugging
        '''

        print(self.weights_vector)
        return self.weights_vector


    def train_on_example(self, training_example: tuple) -> None:
        '''
        This function takes in a training_example and adjusts the weights_vector based on whether or not
        it misclassifies the example.
        '''

        training_vector = training_example[0]
        training_annotation = training_example[1]
        result = self.forward(training_example)
        prediction = self.class_prediction(result)

        #let Class 1 be denoted with the label 0, and Class 2 be denoted with the label 1
        #Additionally let W^T*x > 0 be the classification for Class 1, and W^{T}x <= 0 be the classification
        # for Class 2
        delta_w = (np.zeros(self.input_length + 1))
        if (prediction == 0) and (training_annotation == 1):
            delta_w = (-1)*self.learning_rate*self.process_training_vector(training_vector) 
        elif (prediction == 1) and (training_annotation == 0):
            delta_w = self.learning_rate*self.process_training_vector(training_vector)
        else:
            #Do nothing
            pass

        #Now let's change our vector
        self.weights_vector = np.add(self.weights_vector, delta_w)


    def train_on_dataset(self, training_dataset: tuple):
        '''
        Trains the Perceptron on the entirety of the training_dataset
        '''

        for index in range(len(training_dataset[0])):
            training_vector = training_dataset[0][index]
            training_annotation = training_dataset[1][index]
            training_example = (training_vector, training_annotation)
            self.train_on_example(training_example)


    
    def evaluate_performance(self, test_dataset: tuple) -> float:
        '''
        Evaluates the performance of the machine learning model on the test_dataset. Returns the 
        accuracy by calculating the number of missclassified occurences in the dataset and prints 
        out the results in a string.
        '''

        #Loop through the dataset and calculates the number of missclassifications
        num_wrong = 0
        for index in range(len(test_dataset[0])):
            training_vector = test_dataset[0][index]
            training_annotation = test_dataset[1][index]
            training_example = (training_vector, training_annotation)
            result = self.forward(training_example)
            if (self.class_prediction(result) != training_annotation):
                num_wrong += 1
            
        #Now let's get the results
        accuracy = float(self.training_length - num_wrong) / float(self.training_length)
        print(f"The Perceptron has an Accuracy of {accuracy:.2f}".format(accuracy))

