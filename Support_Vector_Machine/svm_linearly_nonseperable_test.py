import numpy as np
from svm_linearly_nonseperable import SVM_Linearly_NonSeperable
import time

#First let's set up an easy dataset
training_vectors = np.array([
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0])
]
)

training_annotations = np.array([-1.0, 1.0])

training_dataset = (training_vectors, training_annotations)

SVM = SVM_Linearly_NonSeperable(0.001, 0.1, training_dataset)

while True:
    SVM.train()
    SVM.calc_accuracy()
    SVM.calc_hinge_loss()
    time.sleep(1)