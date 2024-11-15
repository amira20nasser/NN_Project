"""
*Perceptron Implementation*
- Implement the Perceptron learning algorithm with the ability to classify data for any two selected classes and two selected features.
- Check Lab3 Task description for steps Perceptron
"""
# Feel free to add new param or functions 
# this is only my insights that we need for algo  

import numpy as np 

class Perceptron:
    def __init__(self, learning_rate, epochs, bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.bias = 0.1* np.random.randn() if bias == True else None
        self.weights = 0.1* np.random.randn(1, 2)
    
    
    def train(self,X,Y):
        X = X.values
        Y=Y.flatten()
        y_pred=np.empty(len(Y))

        print("In Training Perceptron: ",X.shape,Y.shape)
        #  (1,2) . (2 features , training samples ) = (1, training samples)
        for i in range(self.epochs):
            for j in range(len(X)):
                if self.bias is not None:
                      net = np.dot(self.weights, X[j].T) + self.bias
                else:
                    net = np.dot(self.weights, X[j].T)
                y_pred[j] = self.sgn_activation_function(net)
                error = Y[j] - y_pred[j]
                # if np.all(error == 0):
                #     continue
                #                  (1,training samples).(traniing samples,2features)  
                self.weights = self.weights + self.learning_rate * error * X[j]
                if self.bias is not None:
                    self.bias += self.learning_rate * error

    def sgn_activation_function(self,net):
        return np.where(net >= 0, 1, -1)

    def predict(self,X):
        # Write Here YOUR CODE
        if self.bias is not None:
            net = np.dot(self.weights,X.values.T) + self.bias
        else:
            net = np.dot(self.weights,X.values.T) 
        y_pred = self.sgn_activation_function(net)
        return y_pred
        

