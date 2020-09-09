import numpy as np
from scipy.io import savemat, loadmat
from DataSplitter import parse_data
import matplotlib.pyplot as plt


def relu(Z):
    return np.maximum(0, Z)


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


class Model:
    def __init__(self, layer_dims: [int], seed=1):
        np.random.seed(seed)  # Seed for random numbers generation
        self.L = len(layer_dims)  # Number of layers
        self.layer_dims = layer_dims  # Dimensions of each layer
        self.caches = [[] for _ in range(self.L - 1)]
        self.W = [0] * (self.L - 1)
        self.b = [0] * (self.L - 1)
        self.dA = [0] * (self.L - 1)
        self.db = [0] * (self.L - 1)
        self.dW = [0] * (self.L - 1)
        self.costs = []
        self.cost = 0

    def fit(self, folds_X, folds_Y, learning_rate=0.01, number_iterations=6000):
        """" Trains the model. """
        for i in range(number_iterations):
            AL = self.propagate_forward(folds_X)
            self.cost = self.compute_cost(AL, folds_Y)
            self.costs.append(self.cost)
            self.propagate_backward(AL, folds_Y)
            self.update_parameters(learning_rate)
            self.save_weights()
            if i % 50 == 0:
                print("Iterations: ", i, "cost: ", self.cost)
                plt.plot(range(i), self.costs)
                plt.savefig('learning_fig.png')
                self.save_weights()

    def save_weights(self, file='data/weights/weights.mat'):
        savemat(file, mdict={'W': self.W,
                             'b': self.b})

    def load_weights(self, file='data/weights/weights.mat'):
        mat = loadmat(file)
        self.W = mat['W'][0]
        self.b = mat['b'][0]

    def initialize_parameters(self):
        """ Initialize W, b parameters with Xavier initialization. """
        for i in range(1, self.L):
            self.W[i - 1] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * 0.01
            self.b[i - 1] = np.zeros((self.layer_dims[i], 1))

    def propagate_forward(self, X):
        """" Implements forward propagation and returns AL - vector of predictions for samples in X. """

        A = X
        L = self.L - 1

        for i in range(0, L - 1):
            Z = np.matmul(self.W[i], A) + self.b[i]
            cache = ((A, self.W[i], self.b[i]), Z)
            A = relu(Z)
            self.caches[i] = cache
        Z = np.matmul(self.W[L - 1], A) + self.b[L - 1]
        cache = ((A, self.W[L - 1], self.b[L - 1]), Z)
        AL = sigmoid(Z)
        self.caches[L - 1] = cache
        return AL

    def compute_cost(self, AL, Y):
        """ Computes the cross-entropy cost based on predictions vector AL and true labels Y. """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)

        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) * (1 / m)
        db = np.sum(dZ, axis=1, keepdims=True) * (1 / m)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def propagate_backward(self, AL, Y):
        """ Competes gradients for further parameters update. """
        L = len(self.caches)
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        cache = self.caches[L - 1]
        self.dA[L - 1], self.dW[L - 1], self.db[L - 1] = self.linear_backward(sigmoid_backward(dAL, cache[1]), cache[0])
        for i in reversed(range(L - 1)):
            cache = self.caches[i]
            self.dA[i], self.dW[i], self.db[i] = self.linear_backward(sigmoid_backward(self.dA[i + 1], cache[1]),
                                                                      cache[0])

    def update_parameters(self, learning_rate):
        """ Updates the W, b parameters. """
        for i in range(self.L - 1):
            self.W[i] -= learning_rate * self.dW[i]
            self.b[i] -= learning_rate * self.db[i]


if __name__ == '__main__':
    X, Y = parse_data('data/frogs', 'data/toads', save=False)
    X = X[:500].T
    model = Model([X.shape[0], 100, 100, 50, 1])
    model.initialize_parameters()
    Y = Y[:500].reshape((1, Y[:500].shape[0]))
    model.fit(X, Y)