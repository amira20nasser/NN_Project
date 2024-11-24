import numpy as np

class BackPropagation:
    def __init__(self, learning_rate,layers, epochs,neurons,isBias=False,isSigmoid = True):
        self.learning_rate = learning_rate  
        self.layers = layers
        self.epochs = epochs  
        self.isBias = isBias
        self.isSigmoid = isSigmoid
        self.neurons = neurons # list [3] or [4 3] layer 1 layer 2
        self.z=[]
        self.output=[]
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

    def get_sizes(self,X,Y):
        # X shape(90, 5)
        self.input_size  = X.shape[1]
        self.output_size = Y.shape[1]
        self.hidden_size = self.neurons

    def initialize_params(self, X, Y):
        self.weights = [0.01 * np.random.randn(self.hidden_size[0], self.input_size)]
        if self.isBias == True:
            self.bias = [0.01* np.random.randn(1, self.hidden_size[0])]
        else:
            self.bias=0
        i = 1
        for i in range(self.layers):
            self.weights.append( 0.01* np.random.randn(self.hidden_size[i],self.hidden_size[i-1]))  
            if self.isBias == True:
                self.bias.append(0.01* np.random.randn(1, self.hidden_size[i]) )  

        self.weights_output = 0.01* np.random.randn(self.output_size ,self.hidden_size[self.layers - 1])
        self.bias_output = 0.01* np.random.randn(1,self.output_size )
        # print("hidden weights shape", len(self.weights) )
        # print( self.weights[1])
        # print("all hidden BIAS shape",len(self.bias))
        # print( self.bias[0])
        # print("all output weights shape",len(self.weights_output))
        # print( self.weights_output)
        # print("all output BIAS shape",len(self.bias_output))
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
    return 1 - tanh_Z**2


def forward(self, X):
    """ 
    NOTE:
    make custom function that calculate forwad prob for one layer
    note we will call that function multiple times until end of each layer 
    Don't understand ask me 😊

    Feel free to edit that part params ....

    weights is list of arrays  to access first array weights[0]
    weights_output 
    bias 
    bias_output
    """
    print("Forward prop!")
    for i in range(len(self.layers)):
    self.z[i]=np.matmul(X,self.weights[layer])+self.bias
    if self.isSigmoid:
        self.output[i]= sigmoid(z)
    else:
        self.output[i]= tanh(z)
    return output


def backward(self, X, y):
    print()

# not completed
def train(self,X,Y):
    self.get_sizes(X,Y)
    self.initialize_params(X, Y)
    self.forward(X )
    for i in range(self.layers):
        self.backward(X, y)


