import numpy as np
from sklearn.metrics import mean_squared_error

"""
Adaline Implementation
 - Implement the Perceptron learning algorithm with the ability to classify data for any two selected classes and two selected features.
- Check Lab3 Task description for steps Perceptron
"""
# Feel free to add new param or functions 
# this is only my insights that we need for algo  
class Adaline:
    def __init__(self, learning_rate, epochs, mse_threshold, bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = np.random.randn(1,1) if bias==True  else  None
        self.weights = np.random.randn(1,2)
        self.mse_threshold = mse_threshold
    

    def train(self,X,Y):
        # WRITE HERE YOUR CODE
        # Dimentions
        #  (1,2) . (2 features , training samples ) = (1, training samples)
        while(True):
            if self.bias != None:
                net = np.dot(self.weights,X) + self.bias
            else:
                net = np.dot(self.weights,X) 
            y_pred = net
            
            error = Y - y_pred
            mse = mean_squared_error(Y, y_pred)
            if mse <= self.mse_threshold:
                break
            self.weights = self.weights - self.learning_rate * error *  X
            if self.bias != None:
                self.bias = self.bias - self.learning_rate * error * X



    def predict(self,X):
        # Write Here YOUR CODE
        if self.bias != None:
            net = np.dot(self.weights,X) + self.bias
        else:
            net = np.dot(self.weights,X) 
        return net