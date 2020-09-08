import numpy as np
from scipy.io import savemat, loadmat


def relu(x):
    return x, np.maximum(0, x)


def sigmoid(x):
    return x, 1 / (1 + np.exp(-x))


class Model:
    def __init__(self, layer_dims: [int], seed=1):
        np.random.seed(seed)  # Seed for random numbers generation
        self.L = len(layer_dims)  # Number of layers
        self.layer_dims = layer_dims  # Dimensions of each layer
        self.caches = [[] for _ in range(self.L)]
        self.W = [0] * (self.L - 1)
        self.b = [0] * (self.L - 1)
        self.dA = []
        self.db = []
        self.dW = []
        self.cost = 0

    def save_weights(self, file='data/weights/weights.mat'):
        savemat(file, mdict={'W': self.W,
                             'b': self.b})

    def load_weights(self, file='data/weights/weights.mat'):
        mat = loadmat(file)
        self.W = mat['W'][0]
        self.b = mat['b'][0]

    def fit(self, folds_X, folds_Y, learning_rate, number_iterations):
        """" Trains the model. """
        pass

    def initialize_parameters(self):
        """ Initialize W, b parameters with Xavier initialization. """
        for i in range(1, self.L):
            self.W[i - 1] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * \
                            np.sqrt(2 / self.layer_dims[i - 1])
            self.b[i - 1] = np.zeros((self.layer_dims[i], 1))

    def propagate_forward(self, X):
        """" Implements forward propagation and returns AL - vector of predictions for samples in X. """

        A = X
        L = self.L - 1

        for i in range(0, L - 1):
            Z = np.matmul(self.W[i], A) + self.b[i]
            *cache, A = (A, self.W[i], self.b[i]), *relu(Z)
            self.caches[i] = cache

        Z = np.matmul(self.W[L - 1], A) + self.b[L - 1]
        *cache, AL = (A, self.W[L - 1], self.b[L - 1]), *sigmoid(Z)
        self.caches[L - 1] = cache

        return AL

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
    model = Model([500, 4, 3, 1])
    model.initialize_parameters()
    print(model.W)
