import numpy as np


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
        pass

    def propagate_forward(self, X):
        pass

    def compute_cost(self, AL, Y):
        pass

    def propagate_backward(self, AL, Y, caches):
        pass

    def update_parameters(self, learning_rate):
        pass


if __name__ == '__main__':
    pass
