import numpy as np

class SVM_Linearly_NonSeperable():


    def __init__(self, learning_rate: float , C: float, training_data: np.ndarray):
        '''
        This implementation of the SVM_Linearly_NonSeperable class is an SVM Machine without kernelization
        which utilizes slack variables to find the optimal hyper-plane that cuts through a non-linearly separable
        dataset.

        The labels of the dataset should be -1, and 1. Additionally this implementation of the SVM utilizes normal
        Gradient Descent, improvements can be made by either implementing this using SGD, or some sort of Convex 
        Optimization/ Quadratic Programming using both the Primal and Dual Formulations of the SVM Optimization Problem.

        The equation we are trying to minimize is 

            min       psi(w, sigma) =  1/2 (w^{t}w) + C*(sigma_1 + .... + sigma_n)
         w, sigma

         with respect to the conditions y_i(w^{t}x_i + b) >= 1 - sigma_i and sigma-i >= 0 for all i


         This is actually an unconstrained, convex optimization problem, so to solve it we merely can take the 
         gradients of w and sigma with respect to psi and perform gradient descent on this cost function.

        we use this to find the w and b which give us the optimal hyperplane
        '''
        self.learning_rate = learning_rate
        self.C = C
        self.training_data = training_data
        #initialize w to be the zero vector of length n, where n is the number of training examples
        #This is wrong
        self.w = np.zeros(len(training_data[0][0]))
        self.b = 0


    def get_training_example_i(self, index: int) -> tuple:
        '''
        '''

        training_vectors = self.training_data[0]
        training_annotations = self.training_data[1]

        return (training_vectors[index], training_annotations[index])
    

    def calc_sigma_i(self, index: int) -> float:
        '''
        '''

        x_i, y_i = self.get_training_example_i(index)

        #Now we need to calculate sigma_i
        pred = 1 - y_i*(self.w.dot(x_i) + self.b)
        return np.max(pred, 0)
    

    def train(self) -> None:
        '''
        '''

        delta_w = self.w 
        delta_b = 0

        #now we loop through our training_data
        #we do this to calculate delta_w and delta_b
        for index in range(len(self.training_data[0])):

            #Load some things
            x_i, y_i = self.get_training_example_i(index) 
            sigma_i = self.calc_sigma_i(index)

            #Now let's calculate our delta_w 
            if sigma_i == 0:
                delta_w += np.zeros(len(self.training_data[0][0]))
                delta_b += 0
            else:
                delta_w += self.C*((-1)*y_i*x_i)
                delta_b += self.C*((-1)*y_i)

        # now we can use the gradient descent algorithm to update our w and b values
        self.w = self.w - self.learning_rate*delta_w
        self.b = self.b - self.learning_rate*delta_b


    def calc_hinge_loss(self) -> float:
        '''
        '''

        regularizer = 0.5* self.w.dot(self.w)
        hinge_loss = 0

        #now we must loop through our dataset and calculate the hinge_loss
        for index in range(len(self.training_data[0])):
            sigma_i = self.calc_sigma_i(index)
            hinge_loss += self.C*sigma_i

        #Now let's dislay the cost
        loss = regularizer + hinge_loss
        print(f"The Cost of the Current Decision Boundary is {loss:.2f}".format(loss))
        return loss
    

    def calc_accuracy(self):
        '''
        '''

        num_wrong_classified = 0
        total_num = len(self.training_data[0])
        for index in range(len(self.training_data[0])):
            sigma_i = self.calc_sigma_i(index)
            #we can use the value of sigma_i to determine if our model missclassifies or not
            if sigma_i == 0:
                #do nothing
                pass
            else:
                num_wrong_classified += 1 

        #now let's return the accuracy
        accuracy = float(total_num - num_wrong_classified) / float(total_num)
        print(f"The Accuracy of the SVM Algorithm on the Training_Set is {accuracy:.2f}".format(accuracy))