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


        self.bias = 0.1* np.random.randn() if bias == True else None
        self.weights = 0.1* np.random.randn(1, 2)
        self.mse_threshold = mse_threshold

    def train(self, X, Y):
        X = X.values
        Y=Y.flatten()
        y_pred=np.empty(len(Y))

        # Dimentions
        #  (1,2) . (2 features , training samples ) = (1, training samples)
        for i in range(self.epochs):
            for j in range(len(X)):
                if self.bias is not None:
                    net = np.dot(self.weights, X[j].T) + self.bias
                else:
                    net = np.dot(self.weights, X[j].T)
                y_pred[j] = net
                error = Y[j] - y_pred[j]
                # (1,training samples).(traniing samples,2features)  
                self.weights = self.weights + self.learning_rate * error * X[j]
                if self.bias is not None:
                    self.bias += self.learning_rate * error

            mse = mean_squared_error(Y, y_pred)
            # print(f"ADA current mse {mse}")
            if mse <= self.mse_threshold:
                print("ADA Reach less than threshold")
                break
    def predict(self, X):
        # Write Here YOUR CODE
        if self.bias is not None:
            net = np.dot(self.weights, X.values.T) + self.bias
        else:
            net = np.dot(self.weights, X.values.T)
        return np.where(net >= 0, 1, -1).reshape(len(X))

