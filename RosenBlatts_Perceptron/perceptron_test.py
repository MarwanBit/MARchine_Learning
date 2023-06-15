#Here we create some tests for the perceptron test
from perceptron import RosenBlott_Perceptron
import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import time

def test_forward_1():
    '''
    '''

    training_example = (np.array([1,1]), 0)
    weights_vector = np.array([0,0,0])
    perceptron = RosenBlott_Perceptron(2, weights_vector, learning_rate=1.0, training_length=2000)
    #Now let's test if the perceptron returns the correct value
    #This should be [0, 0, 0]* [1, 1, 1] = 0
    result = perceptron.forward(training_example)
    assert result == 0
    print("test_forward_1 successful!")

def test_forward_2():
    '''
    '''

    training_example = (np.array([1,1]), 0)
    weights_vector = np.array([2,3,4])
    perceptron = RosenBlott_Perceptron(2, weights_vector, learning_rate=1.0, training_length=2000)
    result = perceptron.forward(training_example)
    assert result == 9
    print("test_forward_2 successful!")


def vector_test():
    '''
    '''
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 2, 1])
    result = np.add(v1, (-1.0)*v2)
    print(result)

    print("vector_test successful!")


def test_process_training_vector():
    '''
    '''
    v1 = np.array([1, 1])
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=2000)
    v1 = perceptron.process_training_vector(v1)
    print(v1)
    print(v1.dot(np.array([-1, -2, 0.0])))
    print("test_process_training_vector successful!")
    return v1
    

def test_training_on_dataset():
    '''
    '''

    #First let's create an example problem who's solution hyperplane should be y=x with the learning_rate
    # of 1.0
    training_vectors = np.array([
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([2, 1]),
        np.array([3, 1])
    ])
    training_annotations = np.array([0,0,1,1])
    #combine the annotations and the labels into a dataset
    training_dataset = (training_vectors, training_annotations)

    #now let's generate our perceptron
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=4)
    
    perceptron.evaluate_performance(training_dataset)
    #now let's do these operations manually!
    print("Below is W_0")
    perceptron.print_weights()
    for index in range(4):
        training_vector = training_vectors[index]
        training_annotation = training_annotations[index]
        training_example = (training_vector, training_annotation)
        perceptron.train_on_example(training_example)

        print("above is the delta_w for this training example, below is the new weight")
        weights = perceptron.print_weights()

    perceptron.evaluate_performance(training_dataset)

    print("test_training_on_dataset successful!")


def test_misclassification():
    '''
    '''
     #First let's create an example problem who's solution hyperplane should be y=x with the learning_rate
    # of 1.0
    training_vectors = np.array([
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([2, 1]),
        np.array([3, 1])
    ])
    training_annotations = np.array([0,0,1,1])
    #combine the annotations and the labels into a dataset
    training_dataset = (training_vectors, training_annotations)

    #now let's generate our perceptron
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=4)

    #Now let's test the classification
    training_example_0 = (training_vectors[0], training_annotations[0])
    perceptron.misclassification(training_example_0)

    training_example_1 = (training_vectors[1], training_annotations[1])
    perceptron.misclassification(training_example_1)

    training_example_2 = (training_vectors[2], training_annotations[2])
    perceptron.misclassification(training_example_2)

    #return new vector 
    delta_1 = (-1)*perceptron.learning_rate*perceptron.process_training_vector(training_example_2[0])
    print((-1)*perceptron.learning_rate*perceptron.process_training_vector(training_example_2[0]))

    #now let's update this vector
    perceptron.weights_vector = np.add(perceptron.weights_vector, delta_1)
    perceptron.print_weights()

    training_example_3 = (training_vectors[3], training_annotations[3])
    perceptron.misclassification(training_example_3)

    #now let's loop over
    perceptron.misclassification(training_example_0)
    delta_2 = perceptron.learning_rate*perceptron.process_training_vector(training_example_0[0])
    perceptron.weights_vector = np.add(perceptron.weights_vector, delta_2)
    perceptron.print_weights()

    #now let's evaluate the performance
    perceptron.evaluate_performance(training_dataset)


    print("test_missclassification successful!")


def test_misclassification_2():
    '''
    '''
     #First let's create an example problem who's solution hyperplane should be y=x with the learning_rate
    # of 1.0
    training_vectors = np.array([
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([2, 1]),
        np.array([3, 1])
    ])
    training_annotations = np.array([0,0,1,1])
    #combine the annotations and the labels into a dataset
    training_dataset = (training_vectors, training_annotations)

    #now let's generate our perceptron
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=4)

    #Now let's test the classification
    training_example_0 = (training_vectors[0], training_annotations[0])
    perceptron.misclassification(training_example_0)

    training_example_1 = (training_vectors[1], training_annotations[1])
    perceptron.misclassification(training_example_1)

    training_example_2 = (training_vectors[2], training_annotations[2])
    perceptron.misclassification(training_example_2)

    #return new vector 
    perceptron.train_on_example(training_example_2)
    perceptron.print_weights()

    training_example_3 = (training_vectors[3], training_annotations[3])
    perceptron.misclassification(training_example_3)

    #now let's loop over
    perceptron.misclassification(training_example_0)
    perceptron.train_on_example(training_example_0)
    perceptron.print_weights()

    #now let's evaluate the performance
    perceptron.evaluate_performance(training_dataset)


    print("test_missclassification_2 successful!")


def test_misclassification_3():
    '''
    '''
     #First let's create an example problem who's solution hyperplane should be y=x with the learning_rate
    # of 1.0
    training_vectors = np.array([
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([2, 1]),
        np.array([3, 1])
    ])
    training_annotations = np.array([0,0,1,1])
    #combine the annotations and the labels into a dataset
    training_dataset = (training_vectors, training_annotations)

    #now let's generate our perceptron
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=4)

    #Now let's test the classification
    training_example_0 = (training_vectors[0], training_annotations[0])
    perceptron.train_on_example(training_example_0)
    perceptron.print_weights()

    training_example_1 = (training_vectors[1], training_annotations[1])
    perceptron.train_on_example(training_example_1)
    perceptron.print_weights()

    training_example_2 = (training_vectors[2], training_annotations[2]) 
    perceptron.train_on_example(training_example_2)
    perceptron.print_weights()

    training_example_3 = (training_vectors[3], training_annotations[3])
    perceptron.train_on_example(training_example_3)
    perceptron.print_weights()

    #now let's loop over
    perceptron.train_on_example(training_example_0)
    perceptron.print_weights()

    #now let's evaluate the performance
    perceptron.evaluate_performance(training_dataset)


    print("test_missclassification_3 successful!")


def test_misclassification_4():
    '''
    '''
     #First let's create an example problem who's solution hyperplane should be y=x with the learning_rate
    # of 1.0
    training_vectors = np.array([
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([2, 1]),
        np.array([3, 1])
    ])
    training_annotations = np.array([0,0,1,1])
    #combine the annotations and the labels into a dataset
    training_dataset = (training_vectors, training_annotations)

    #now let's generate our perceptron
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=4)

    #Now let's test the classification
    perceptron.train_on_dataset(training_dataset)
    perceptron.evaluate_performance(training_dataset)
    perceptron.train_on_dataset(training_dataset)
    perceptron.evaluate_performance(training_dataset)


    print("test_missclassification_3 successful!")


if __name__ == "__main__":

    #first let us define a general purpose perceptron and training_example
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=2000)
    training_example = (np.array([1,1]), 0)

    #Test the forward method of the perceptron
    test_forward_1()
    test_forward_2()
    print("\n")

    #now let's do the vector test!
    vector_test()
    test_process_training_vector()
    print("\n")

    #now let's test when we get misclassifications
    test_misclassification()
    print("\n")
    test_misclassification_2()
    print("\n")
    test_misclassification_3()
    print("\n")
    test_misclassification_4()
    print("\n")

    
    separable = False
    samples = 0
    #This code creates random datasets of 1 clump per class, checks if it is seperable and continues
    #until a separable dataset is generated
    while not separable:
        samples = make_classification(n_samples=2000, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
 
    #Now let's visualize our dataset
    plt.plot(red[:, 0], red[:, 1], 'r.')
    plt.plot(blue[:, 0], blue[:, 1], 'b.')
    plt.show()

    #split it into the actual datapoints and the class labels
    dataset_points = samples[0]
    dataset_labels = samples[1]
    
    #Now let's train our perceptron on this dataset
    while True:
        perceptron.train_on_dataset(samples)
        perceptron.evaluate_performance(samples)

        #Now let's display the results
        #Now let's visualize our dataset
        plt.plot(red[:, 0], red[:, 1], 'r.')
        plt.plot(blue[:, 0], blue[:, 1], 'b.')

        #here let's get the equation of our line
        w_0 = perceptron.weights_vector[0]
        w_1 = perceptron.weights_vector[1]
        w_2 = perceptron.weights_vector[2]

        #now we can calculate the slope and intercept of our line
        slope = -(w_0/w_2)/(w_0/w_1)  
        intercept = -w_0/w_2

        #now we iterate and plot some lines
        for i in np.linspace(-2.0, 2.0, 50):
            y = slope*i + intercept 
            plt.plot(i, y, 'ko')

        plt.show()

        time.sleep(1)