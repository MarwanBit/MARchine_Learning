import numpy as np
import cvxopt

class SVM_Kernel():
    def __init__(self, learning_rate: float, C: float,  kernel_function, training_data: tuple):
        '''
        This implements the non-linear SVM with kernelization. The way this works is we do the 
        optimization with the lagrange multipliers, making sure to use gradient descent to solve
        the convex optimization problem. What we then do is use this to calculate the decision boundary
        and use this to classify the dataset.

        Here are the steps that we will follow in order to generate our SVM with Kernelization.
         
        Notice for this implementation we don't calculate the weight vector explicitly,we use the kernel trick for 
        classification
        '''
        self.lr = learning_rate 
        self.C = C
        self.kernel = kernel_function 
        self.kernel_matrix = 0
        self.training_data = training_data 
        self.sv = 0


    def get_training_example_i(self, index: int) -> tuple:
        '''
        '''
        training_vectors = self.training_data[0]
        training_annotations = self.training_data[1]
        return (training_vectors[index], training_annotations[index])
    

    def train(self) -> np.array:
        '''
        '''
        n_samples = len(self.training_data[0])
        #now we must construct our gram_matrix 
        self.kernel_matrix = np.zeros((n_samples, n_samples))
        d_vector = self.training_data[1]
        print("Shape of Training Dataset followed by the shape of the training annotations")
        print(self.training_data[0].shape)
        print(d_vector.shape)
        print("constructing Kernel Matrix")
        for i in range(n_samples):
            for j in range(n_samples):
                x_i = self.get_training_example_i(i)[0]
                x_j = self.get_training_example_i(j)[0]
                self.kernel_matrix[i][j] = self.kernel(x_i, x_j)
        print("Done with Kernel Matrix")
        # now let's construct the matrices for the convex optimization problem
        print("Constructing Quadratic Optimization Matrices")
        print("Size of d^td: " + str(np.outer(d_vector, d_vector).shape))
        P = cvxopt.matrix(np.outer(d_vector, d_vector)* self.kernel_matrix)
        print("Size of Kernel_Matrix: " + str(self.kernel_matrix.shape))
        print("Constructed P")
        q = cvxopt.matrix(np.ones(n_samples) * (-1))
        print("Constructed q")
        A = cvxopt.matrix(d_vector, (1, n_samples), 'd')
        print("Constructed A")
        b = cvxopt.matrix(0.0)
        print("Constructed b")
        # now we construct our G and h matrices for the condition G(alpha) <= h
        #these create the conditions 0 <= alpha_i <= C 
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
        print("Constructed G")
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        print("Constructed h")
        #now we solve the optimization problem
        print("solving optimization problem")
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        print("")
        # now we get our solution vector
        a = np.ravel(solution['x'])
        self.lagrange_multipliers = a
        self.sv = a > 1e-5 # some small threshold
        return self.sv
    
    
    def res(self, index) -> float:
        '''
        ''' 
        x_i, y_i = self.get_training_example_i(index)  
        #we calculate w_dot_x using the statement that w*phi(x) = \sum
        w_dot_phix  = 0 
        for j_index, val in enumerate(self.sv):
            #check if the index is apart of the support vector
            if val:
                #calculates sum_{1 <= j <= m}alpha_j*y_j*K(x__j, x_i)
                alpha_j = self.lagrange_multipliers[j_index]
                x_j, y_j = self.get_training_example_i(j_index)
                w_ker_val = self.kernel_matrix[j_index][index]
                w_dot_phix += alpha_j*y_j*w_ker_val
        #returns the sign of the function
        return w_dot_phix
    
    
    def classify(self, index) -> None:
        '''
        '''
        res = self.res(index)
        return np.sign(res)


    def get_accuracy(self) -> None:
        '''
        '''
        num_wrong_classified = 0 
        for index in range(len(self.training_data[0])):
            x_index, y_index = self.get_training_example_i(index)
            predicted_class = self.classify(index)
            if (predicted_class != float(y_index)):
                num_wrong_classified += 1 
        #Now we calculate the accuracy
        total_num_samples = len(self.training_data[0])
        accuracy = float(total_num_samples - num_wrong_classified) / float(total_num_samples)
        print(f"The accuracy of the SVM is : {accuracy:.2f}".format(accuracy))