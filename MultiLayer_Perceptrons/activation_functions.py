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

# Cross Entropy function.
def cross_entropy(y_pred, y_true):
 
    # computing softmax values for predicted values
    y_pred = softmax(y_pred)
    loss = 0
     
    # Doing cross entropy Loss
    for i in range(len(y_pred)):
 
        # Here, the loss is computed using the
        # above mathematical formulation.
        loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))
 
    return loss

def cross_entropy_prime(y_pred, y_true):
    vec = [[((-1)*y_pred[i][0] / y_true[i][0])] for i in range(len(y_pred))]
    return np.array(vec)