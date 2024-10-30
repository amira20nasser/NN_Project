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
        self.bias = np.random.randn(1,1) if bias==True  else  None
        self.weights = np.random.randn(1,2) 
    
    
    def train(self,X,Y):
        # WRITE HERE YOUR CODE
        # Dimentions
        #  (1,2) . (2 features , training samples ) = (1, training samples)
        while(True):
            if self.bias != None:
                net = np.dot(self.weights,X) + self.bias
            else:
                net = np.dot(self.weights,X) 
            y_pred = sgn_activation_function(net)
            error = Y - y_pred
            if error == np.zeros((1,X.shape[1])):
                break
            # (1,2)                   (1,training examples) * (2,training examples)
            self.weights = self.weights - self.learning_rate * error *  X
            if self.bias != None:
                self.bias = self.bias - self.learning_rate * error * X

    def sgn_activation_function(net):
        return 1 if net >= 0 else -1

    def predict(self,X):
        # Write Here YOUR CODE
        if self.bias != None:
            net = np.dot(self.weights,X) + self.bias
        else:
            net = np.dot(self.weights,X) 
        y_pred = sgn_activation_function(net)
        return y_pred
        

