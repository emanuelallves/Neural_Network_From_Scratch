import numpy as np
from core.activation import Activation

class Layer:
    def __init__(self, input_size, output_size, activation="ReLU"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.activation = Activation.get(self.activation_name)

        weights_limit = np.sqrt(6 / (input_size + output_size))

        self.weights = np.random.uniform(-weights_limit, weights_limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size))

        self.input_data = None
        self.z = None
        self.a = None

    def forward(self, inputs):
        self.input_data = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation(self.z)

        return self.a

    def backward(self, dA, lr):
        activation_derivative = Activation.derivative(self.activation_name, self.z)
        dZ = dA * activation_derivative
        dW = np.dot(self.input_data.T, dZ) / self.input_data.shape[0]
        dB = np.mean(dZ, axis=0, keepdims=True)

        dA_prev = np.dot(dZ, self.weights.T)

        self.weights -= lr * dW
        self.bias -= lr * dB

        return dA_prev