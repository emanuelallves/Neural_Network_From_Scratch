import numpy as np

class Activation:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        tanh = Activation.tanh(x)
        return 1 - tanh ** 2

    @staticmethod
    def get(name):
        activations = {
            "sigmoid": Activation.sigmoid,
            "ReLU": Activation.ReLU,
            "tanh": Activation.tanh,
        }

        return activations[name]

    @staticmethod
    def derivative(name, x):
        activations = {
            "sigmoid": Activation.sigmoid_derivative,
            "ReLU": Activation.ReLU_derivative,
            "tanh": Activation.tanh_derivative,
        }

        return activations[name](x)
