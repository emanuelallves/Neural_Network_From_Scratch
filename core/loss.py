import numpy as np

class Loss:

    @staticmethod
    def MSE(y_true, y_pred):
        loss = np.mean((y_pred - y_true) ** 2)
        gradient = 2 * (y_pred - y_true) / y_true.shape[0]

        return loss, gradient

    @staticmethod
    def CrossEntropy(y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon) #Faz com que o intervalo do y_pred fique entre [10^-12, 1 - 10^-12]
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        gradient = (y_pred / y_true) / y_true.shape[0]

        return loss, gradient

    @staticmethod
    def get(name):
        losses = {
            "MSE": Loss.MSE,
            "CrossEntropy": Loss.CrossEntropy,
        }

        return losses[name]