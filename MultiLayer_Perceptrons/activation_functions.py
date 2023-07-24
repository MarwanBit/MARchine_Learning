import numpy as np
from scipy.special import softmax

def tanh(x):
    return np.tan(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def soft_max_prime(x): 
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (n, 1) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    s = softmax(x)
    jacobian_m = np.zeros((len(s), len(s)))
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i][0] * (1-s[i][0])
            else: 
                jacobian_m[i][j] = -s[i][0] * s[j][0]
    return jacobian_m