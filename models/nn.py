import numpy as np
from models.layer import Layer

class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation="ReLU"):
        self.layers.append(Layer(input_size, output_size, activation))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient, lr):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, lr)

    def train(self, X, y, epochs, lr, loss_function):
        for epoch in range(epochs):
            output = self.forward(X)
            loss, loss_gradient = loss_function(y, output)
            self.backward(loss_gradient, lr)
            print(f"Epoch: {epoch} -> Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)