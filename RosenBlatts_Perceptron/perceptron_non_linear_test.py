from perceptron import RosenBlott_Perceptron 
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import time

#Non-Linear Dataset
X_moon, y_moon = make_moons(
    n_samples=1000,
    shuffle=True,
    random_state=12323,
    noise = 1.0
)

if __name__ == "__main__":
    perceptron = RosenBlott_Perceptron(2, np.array([0,0,0]), learning_rate=1.0, training_length=len(X_moon))
    while True:
        perceptron.train_on_dataset((X_moon, y_moon))
        perceptron.evaluate_performance((X_moon, y_moon))
        time.sleep(0.5)
 