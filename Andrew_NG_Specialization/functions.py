#let's import numpy 
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use("deeplearning.mplstyle")

def compute_model_output(x, w, b):
    '''
    Here's an example computation 

    x = [1.0, 2.0]
    w = 2.0
    b = 1.0 

    y = 2.0* [1.0, 2.0] + 1.0
    y = [2.0, 4.0] + 1.0
    y = [3.0, 5.0]

    '''

    # shape is (# rows, # cols)
    m = x.shape[0]
    out = np.zeros((m))
    for i in range(m):
        out[i] = w*x[i] + b
    return out


# now we can use this to guess and plot our predictions 
def plot_prediction(x_train, y_train, w, b):
    prediction = compute_model_output(x_train, w, b)
    #Now let's plot our predictions 
    plt.plot(x_train, prediction, label = "our prediction", c = "r")
    plt.scatter(x_train, y_train, marker = "x", c= "b", label = "Actual Values")
    plt.title("Housing Prices")
    plt.xlabel("Square Footage")
    plt.ylabel("Price (per 1k)")
    plt.legend()
    plt.show()