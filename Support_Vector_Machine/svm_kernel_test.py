import numpy as np
from svm_kernel import SVM_Kernel
import time
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math

def polynomial_kernel(x_i: np.array, x_j: np.array, p : int = 4.0):
    '''
    '''
    return (1 + x_i.dot(x_j))**p

def linear_kernel(x_i: np.array, x_j: np.array):
    '''
    '''
    return x_i.dot(x_j)

def null_kernel(x_i: np.array, x_j: np.array):
    '''
    '''
    return 0

def GRBF_kernel_generator(sigma: float):
    '''
    '''
    def GRBF_kernel_func(x_i: np.array, x_j: np.array):
        coeff = (-1/ (2*(sigma**2)))
        return math.exp(coeff* np.linalg.norm((x_i - x_j))**2)
    return GRBF_kernel_func


# Creating dataset
X, y = datasets.make_blobs(

        n_samples = 1000, # Number of samples
        n_features = 2, # Features
        centers = 2,
        cluster_std = 1,
        random_state=40
    )

#Non-Linear Dataset
X_moon, y_moon = datasets.make_moons(
    n_samples=1000,
    shuffle=True,
    random_state=12323,
    noise = 1.0
)

#This is for the XOR Problem which we will use for experimentation
X_xor = np.array([
    np.array([-1.0, -1.0]),
    np.array([-1.0, +1.0]),
    np.array([+1.0, -1.0]),
    np.array([+1.0, +1.0]),
])
y_xor = np.array([-1.0, 1.0, 1.0, -1.0])

# Classes 1 and -1
y = np.where(y == 0, -1, 1)
y_moon = np.where(y_moon == 0, -1, 1)
training_dataset_1 = (X, y)
training_dataset_2 = (X_moon, y_moon)
training_dataset_xor = (X_xor, y_xor)

if __name__ == "__main__":
    SVM = SVM_Kernel(0.001, 0.01, GRBF_kernel_generator(sigma=0.01), training_dataset_2)
    SVM.train()

    #let's visualize the dataset
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_xor[:, 0], X_xor[:, 1], marker="o", c=y_xor)
    plt.show()

    print(SVM.kernel_matrix)
    print(SVM.lagrange_multipliers)
    print(SVM.lagrange_multipliers[0])
    print(SVM.lagrange_multipliers[1])
    print(SVM.lagrange_multipliers[2])
    print(SVM.lagrange_multipliers[3])

    #Now let's get the accuracy
    for index in range(len(X_xor)):
        x_index, y_index = SVM.get_training_example_i(index)
        res = SVM.res(index)
        predicted_class = SVM.classify(index)
        print(res)
        print("Predicted Class: " + str(predicted_class) + " : Actual Class: " + str(y_index))
    # now let's print the accuracy
    SVM.get_accuracy()