import numpy as np

class Activation:

    def __init__(self, x):
        self.x = x

    def forward(self):
        return 1 / (1 + np.exp(-self.x))

    def backward(self):
        return self.forward() * (1 - self.forward())



class Loss:

    def __init__(self, guess, truth):
        self.guess = guess
        self.truth = truth

    def binary_cross_entropy(self):
        return np.mean(-self.truth * np.log(self.guess) - (1 - self.truth) * np.log(1 - self.guess))

    def mean_squared_error(self):
        return np.mean((self.guess - self.truth) ** 2)

    def cross_entropy(self):
        return np.mean(-self.truth * np.log(self.guess))

    def binary_cross_entropy_prime(self):
        return ((1 - self.truth) / (1 - self.guess) - self.truth / self.guess) / self.guess.shape[0]
        
       

    