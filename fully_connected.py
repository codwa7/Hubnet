import numpy as np

class FullyConnected:

    def __init__(self, input_size, output_size):

        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)

    def forward(self, X):
        self.X = X
        return np.dot(self.W, self.X) + self.b
    
    def backward(self, output_grad, learning_rate):
        
        self.dW = np.dot(output_grad, self.X.T)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * output_grad
        
        return np.dot(self.W.T, output_grad)
