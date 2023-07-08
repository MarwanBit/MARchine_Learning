import numpy as np
from svm_linearly_nonseperable import SVM_Linearly_NonSeperable
import time
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# Creating dataset
X, y = datasets.make_blobs(

        n_samples = 10000, # Number of samples
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


# Classes 1 and -1
y = np.where(y == 0, -1, 1)
training_dataset_2 = (X, y)
# Classes 1 and -1
y_moon = np.where(y_moon == 0, -1, 1)
training_dataset_moon = (X_moon, y_moon)


#First let's set up an easy dataset
training_vectors = np.array([
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0])
]
)
training_annotations = np.array([-1.0, 1.0])
training_dataset = (training_vectors, training_annotations)



SVM = SVM_Linearly_NonSeperable(0.01, 0.1, training_dataset)

for i in range(10):
    SVM.train()
    SVM.print_weight_vector()
    print(SVM.b)

    print("Training Example 1: " + str(SVM.get_training_example_i(0)[0]) + ": predicted class" + str(SVM.pred_class(0)) + ": actual class " +\
          str(SVM.get_training_example_i(0)[1]))
    print(SVM.compute_res(0))
    
    print("Training Example 2: " + str(SVM.get_training_example_i(1)[0]) + ": predicted class" + str(SVM.pred_class(1)) + ": actual class " +\
          str(SVM.get_training_example_i(1)[1]))
    print(SVM.compute_res(0))
    
          
    SVM.calc_accuracy()
    SVM.calc_hinge_loss()


#let's visualize the dataset
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
plt.show()

for i in range(10):
    SVM = SVM_Linearly_NonSeperable(0.01, 0.1, training_dataset_2)
    SVM.train()
    SVM.calc_accuracy()
    SVM.calc_hinge_loss()

#Finally let's display some of our results to check
for index in range(100):
    print("Predicted Label: " + str(SVM.pred_class(index)) + ": Actual Label: " + str(SVM.get_training_example_i(index)[1]))


#Testing on the non-linear dataset
for i in range(10):
    print("Testing on Make Moons")
    SVM = SVM_Linearly_NonSeperable(0.01, 0.1, training_dataset_moon)
    SVM.train()
    SVM.calc_accuracy()
    SVM.calc_hinge_loss()