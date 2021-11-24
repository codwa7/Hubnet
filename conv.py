import numpy as np
import scipy.signal as sp

class Conv:

    def __init__(self, input_size, kernal_size, d):
        self.input_size = input_size
        self.kernal_size = kernal_size
        self.d = d
        self.construct()
        self.init_rand()

    def construct(self):
        self.input_width = self.input_size[0]
        self.input_height = self.input_size[1]
        self.input_depth = self.input_size[2]

        self.output_dim = [1 + (self.input_width - self.kernal_size), 1 + (self.input_height - self.kernal_size), self.d]
        self.kernal_dim = [self.kernal_size, self.kernal_size, self.d, self.input_depth]

    def init_rand(self):
        self.kernals = np.random.randn(self.kernal_dim[0], self.kernal_dim[1], self.kernal_dim[2], self.kernal_dim[3])
        self.bias = np.random.randn(self.output_dim[0], self.output_dim[1], self.output_dim[2])


    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)
        for i in range(self.d):
            for j in range(self.input_depth):
                self.output[i] += sp.correlate2d(self.input[j], self.kernals[i, j], "valid")
        return self.output

    def backward(self, grad, learning_rate):
        kernal_grad = np.zeros(self.kernal_dim)
        input_grad = np.zeros(self.input_size)

        for i in range(self.output_dim[0]):
            for j in range(self.output_dim[1]):
                for k in range(self.output_dim[2]):
                    input_grad[i:i+self.kernal_dim[0], j:j+self.kernal_dim[1], k:k+self.kernal_dim[2]] += self.kernals * grad[i, j, k]
                    kernal_grad += self.input[i:i+self.kernal_dim[0], j:j+self.kernal_dim[1], k:k+self.kernal_dim[2]] * grad[i, j, k]

        self.kernals -= learning_rate * kernal_grad
        self.bias -= learning_rate * grad
        return input_grad
      