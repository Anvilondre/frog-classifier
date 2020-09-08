import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Model:
    def __init__(self, layer_dims: [int]):
        self.L = len(layer_dims)  # Number of layers
        self.layer_dims = layer_dims  #
        self.W = \
            self.b = \
            self.dA = \
            self.db = \
            self.dW = \
            self.cost = ...

    def fit(self, folds_X, folds_Y, learning_rate, number_iterations):
        pass

    def initialize_parameters(self):
        """ Initialize W, b parameters with Xavier initialization. """
        pass

    def propagate_forward(self, X):
        """" Implements forward propagation and returns AL - vector of predictions for samples in X. """
        pass

    def compute_cost(self, AL, Y):
        """ Computes the cross-entropy cost based on predictions vector AL and true labels Y. """
        pass

    def propagate_backward(self, AL, Y, caches):
        """ Competes gradients for further parameters update. """
        pass

    def update_parameters(self, learning_rate):
        """ Updates the W, b parameters. """
        pass


if __name__ == '__main__':
    pass
