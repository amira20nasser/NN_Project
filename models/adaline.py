import numpy as np


class Adaline:
    def __init__(self, learning_rate, epochs, mse_threshold, bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.bias = np.random.randn(1, 1) if bias == True else None
        self.weights = 0.1 * np.random.randn(1, 2)
        self.mse_threshold = mse_threshold

    def train(self, X, Y):
        X = X.values
        Y = Y.flatten()
        y_pred = np.empty(len(Y))

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
                self.weights = self.weights + self.learning_rate * error * X[j]

                if self.bias is not None:
                    self.bias += self.learning_rate * error
                    net = np.dot(self.weights, X[j].T) + self.bias
                else:
                    net = np.dot(self.weights, X[j].T)
                y_pred[j] = net
            mse = np.mean((Y - y_pred) ** 2)
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
