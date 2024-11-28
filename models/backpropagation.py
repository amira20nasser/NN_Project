import numpy as np


class BackPropagation:
    def __init__(self, learning_rate, layers, epochs, neurons, isBias=False, isSigmoid=True):
        self.learning_rate = learning_rate
        self.layers = layers
        self.epochs = epochs
        self.isBias = isBias
        self.isSigmoid = isSigmoid
        self.neurons = neurons  # list [3] or [4 3] layer 1 layer 2
        self.output_forward = []
        self.weights = [] #amira bthbd wla eh
        self.bias = []
        self.z_values = []

    """
    Thinking 
    we have 
     5 features 
     suppose 2 layers neurons/layer [3 4] 
     layer weight = (current layer neurons , pervious layer neurons)   
     first layer weights (3,5) dot X (90, 5).T  = (3,90)
     second layer weights (4,3) dot (3,90)  = (4,90)    
     output layer wights (3,4) dot (4,90) = (3,90)   
    """

    def get_sizes(self, X, Y):
        # X shape(90, 5)
        self.input_size = X.shape[1]
        self.output_size = Y.shape[1]
        self.hidden_size = self.neurons

    def initialize_params(self, X, Y):
        self.weights = [0.01 * np.random.randn(self.hidden_size[0], self.input_size)]
        if self.isBias == True:
            self.bias = [0.01 * np.random.randn(self.hidden_size[0],1)]
        else:
            self.bias = None

        for i in range(1, self.layers):
            self.weights.append(0.01 * np.random.randn(self.hidden_size[i], self.hidden_size[i - 1]))
            if self.isBias == True:
                self.bias.append(0.01 * np.random.randn(self.hidden_size[i],1))

        self.weights.append(0.01 * np.random.randn(self.output_size, self.hidden_size[self.layers - 1]))  # [[-1]
        self.bias.append(0.01 * np.random.randn(self.output_size,1))
        # print("hidden weights list len", len(self.weights) )
        # print( self.weights)
        # print("all hidden BIAS shape",len(self.bias))
        # print( self.bias)
        # print("all output weights list len",len(self.weights_output))
        # print( self.weights_output)
        # print("all output BIAS list len",len(self.bias_output))
        # print( self.bias_output)

    # implement from Scratch
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        sigmoid_Z = self.sigmoid(Z)
        return sigmoid_Z * (1 - sigmoid_Z)

    def tanh_derivative(self, Z):
        tanh_Z = self.tanh(Z)
        return 1 - tanh_Z ** 2

    def forward(self, X):
        # print("Forward prop!")
        self.z_values = []
        self.output_forward = []
        input_data = X.values.T
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        # print(input_data.shape)   
        # print(self.bias[0].shape) 
        # (3,5)  . (5,1) + (3,1)
        z = np.dot(self.weights[0], input_data) + (self.bias[0] if self.isBias else 0)
        self.z_values.append(z)
        self.output_forward.append(self.sigmoid(self.z_values[0]) if self.isSigmoid else self.tanh(self.z_values[0]))
        input_data = self.output_forward[0] #(3,1)
        # print(self.output_forward[0].shape)
        # print(self.z_values[0].shape)

        for i in range(1, self.layers+1):         #(4,3)      #(3,1) + (4,1) = (4,1)
            z = np.dot(self.weights[i], input_data) + (self.bias[i].reshape(-1, 1) if self.isBias else 0)
            self.z_values.append(z)
            # print(self.weights[i].shape)
            # print(input_data.shape)
            # print(self.bias[i].shape)
            # print(z)
            if self.isSigmoid:
                self.output_forward.append(self.sigmoid(self.z_values[i]))
            else:
                self.output_forward.append(self.tanh(self.z_values[i]))

            input_data = self.output_forward[i]
        
        # for i in range(3):
        #     print("layer", i + 1)
        #     print("weight shape",self.weights[i].shape)
        #     print("Z values layer  shape", self.z_values[i].shape)
        #     print("output_forward shape", self.output_forward[i].shape)


    def backward(self, x, y):
        output = self.output_forward[-1]
        deltas = [0] * (self.layers + 1)
        f_dash = (self.sigmoid_derivative(self.z_values[-1]) if self.isSigmoid else self.tanh_derivative(self.z_values[-1]))
        # print(f_dash.shape)
        # print(y.values.reshape(-1,1).shape)
        deltas[-1] = (y.values.reshape(-1,1) - output) * f_dash
        self.weights[-1] += self.learning_rate * np.dot(deltas[-1], self.output_forward[-2].T)
        self.bias[-1] += self.learning_rate * np.sum(deltas[-1],axis=1, keepdims=True)

        for i in reversed(range(self.layers)):
            # print(i)                 # (3,4) . (3,1)
            # print("weights shape",self.weights[i+1].shape)
            sum_weights = np.dot(self.weights[i+1].T,deltas[i+1])
            # print("sum weights shape",sum_weights.shape)
            deltas[i] =  sum_weights*(self.sigmoid_derivative(self.z_values[i]) if self.isSigmoid else self.tanh_derivative(self.z_values[i]))
           
            # print(self.output_forward[i].reshape(1, -1))
            # print(self.output_forward[i].reshape(-1, 1))
            
            # print("shape",deltas[i].shape,self.output_forward[i-1].T.shape)
            self.bias[i] += self.learning_rate * np.sum(deltas[i],axis=1, keepdims=True)
            if i==0:
                input_data =x.values
                if input_data.ndim == 1:
                    input_data = input_data.reshape(1, -1)
                else:
                    input_data = input_data
                self.weights[i] += self.learning_rate * np.dot(deltas[i], input_data)
                continue
            self.weights[i] += self.learning_rate * np.dot(deltas[i], self.output_forward[i-1].T)
    

        # # print(deltas)
        # print("=======")
        # for i in range(len(self.weights)):
        #     print(self.weights[i].shape)
        #     print(self.bias[i].shape)
        #     print("======")

    def train(self, X, Y):
        self.get_sizes(X, Y)
        self.initialize_params(X, Y)
        m = X.shape[0]
        # self.forward(X.iloc[0,:])
        # self.backward(X.iloc[0,:], Y.iloc[0,:])
        print("Weights Before")
        print(self.weights[0])

        for e in range(self.epochs):
            for i in range(m):
                self.forward(X.iloc[i,:])
                self.backward(X.iloc[i,:], Y.iloc[i,:])
        print("Weights After")
        print(self.weights[0])
    # not sure
    def predict(self,X):
        predictions = []
        # m = X.shape[0]
        self.forward(X)
        # print()
        # print(self.output_forward[-1].shape) #(3,90)
        for i in range(self.output_forward[-1].shape[1]): 
            p = self.output_forward[-1][:,i]
            max_value = max(p)
            predictions.append([1 if value == max_value else 0 for value in p])
        # print(len(predictions))
        # print(predictions[0])
        return predictions
        

