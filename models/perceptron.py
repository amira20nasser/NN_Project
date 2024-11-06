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

        self.bias = np.random.randn() if bias==True  else  None
        self.weights = np.random.randn(1,2)
    
    
    def train(self,X,Y):
        X = X.values.T
        Y = Y.reshape(1,-1)
        print("In Training Perceptron: ",X.shape,Y.shape)
        #  (1,2) . (2 features , training samples ) = (1, training samples)
        while(self.epochs):
            self.epochs -= 1
            net = 0
            if self.bias is not None:
                print("Bias")
                print(self.bias)
                net = np.dot(self.weights,X) + self.bias
            else:
                net = np.dot(self.weights,X) 
            y_pred = self.sgn_activation_function(net)
            error = Y - y_pred
            # if np.all(error == 0):
            #     continue
            #                  (1,training samples).(traniing samples,2features)  
            self.weights = self.weights + self.learning_rate * np.dot(error, X.T)
            if self.bias is not None:
                self.bias = self.bias + self.learning_rate * np.dot(error, X.T)

    def sgn_activation_function(self,net):
        return [1 if n >= 0 else -1 for n in net[0]]

    def predict(self,X):
        # Write Here YOUR CODE
        if self.bias is not None:
            net = np.dot(self.weights,X.values.T) + self.bias
        else:
            net = np.dot(self.weights,X.values.T) 
        y_pred = self.sgn_activation_function(net)
        return y_pred
        

