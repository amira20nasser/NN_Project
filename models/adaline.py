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
        self.mse_threshold = mse_threshold
        self.bias = bias
        self.weights = None
    
    def train(self,X,Y):
        print()
        # WRITE HERE YOUR CODE
    
    def predict(self,X):
        print()
        #write here YOUR CODE
        #note i think it should return y_pred