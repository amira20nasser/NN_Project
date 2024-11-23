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

        self.bias = np.random.randn() if bias == True else None
        self.weights = np.random.randn(1, 2)
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
            if mse <= self.mse_threshold:
                break
            # (1,training samples).(traniing samples,2features)  
            self.weights = self.weights + self.learning_rate * np.dot(error, X.T)
            if self.bias is not None:
                self.bias += self.learning_rate * np.sum(error)

    def predict(self, X):
        # Normalize data with same transformation as training
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X = X.values.T  # Ensure this is in the correct shape for matrix operations

        # Compute net input
        if self.bias is not None:
            net_input = np.dot(self.weights, X) + self.bias
        else:
            net_input = np.dot(self.weights, X)

        # Apply threshold function to classify outputs
        predictions = np.where(net_input >= 0.5, 1, 0)

        return predictions.flatten()

