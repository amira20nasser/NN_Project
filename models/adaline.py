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

        self.bias = np.random.randn(1,1) if bias == True else None
        self.weights = np.random.randn(1,2) * 0.01
        self.mse_threshold = mse_threshold

    def train(self, X, Y):
        X = X.values.T
        Y = Y.reshape(1, -1)
        # Dimentions
        #  (1,2) . (2 features , training samples ) = (1, training samples)
        for _ in range(self.epochs):
            self.epochs -= 1
            net = 0
            if self.bias is not None:
                net = np.dot(self.weights, X) + self.bias
            else:
                net = np.dot(self.weights, X)
            y_pred = net

            error = Y - y_pred
            mse = mean_squared_error(Y, y_pred)
            # print(f"ADA current mse {mse}")
            if mse <= self.mse_threshold:
                print("ADA Reach less than threshold")
                break
            # (1,training samples).(traniing samples,2features)  
            self.weights = self.weights + self.learning_rate * np.dot(error, X.T)
            if self.bias is not None:
                self.bias += self.learning_rate * np.sum(error)
    def predict(self, X):
        # Write Here YOUR CODE
        if self.bias is not None:
            net = np.dot(self.weights, X.values.T) + self.bias
        else:
            net = np.dot(self.weights, X.values.T)
        return np.where(net >= 0, 1, -1)

