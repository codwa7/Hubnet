import numpy as np
from dataload import *
from conv import *
from utils import *
from fully_connected import *



data = get_CIFAR10_data()

X_train, y_train, X_val, y_val, X_test, y_test = data

model = {
    Conv([28, 28, 3], 3, 5)
}

epochs = 10
learning_rate = 0.001

for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        for layer in model:
            layer.forward(x)
            x = layer.output
        loss = cross_entropy(y, x)
        loss.backward()
        for layer in reversed(model):
            layer.backward()
        for layer in model:
            layer.update(learning_rate)
        if i % 100 == 0:
            print("Epoch: {}, Step: {}, Loss: {}".format(epoch, i, loss.data))





