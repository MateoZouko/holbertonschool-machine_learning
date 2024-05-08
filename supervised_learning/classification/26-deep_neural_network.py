#!/usr/bin/env python3
"""
Task 16
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Deep neural network
    """

    def __init__(self, nx, layers):
        """
        class constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights['W1'] = np.random.randn(
                        layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W{}'.format(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        """
        for i in range(self.__L + 1):
            if i == 0:
                self.__cache["A0"] = X
            else:
                n = np.dot(self.__weights["W{}".format(
                    i)], self.__cache["A{}".format(i - 1)]
                    ) + self.__weights["b{}".format(i)]
                sig_n = 1 / (1 + np.exp(-n))
                self.__cache["A{}".format(i)] = sig_n

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the neural network’s predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dz = self.__cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A = self.__cache["A{}".format(i - 1)]
            dW = np.dot(dz, A.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.dot(self.__weights["W{}".format(i)].T, dz) * A * (1 - A)
            self.__weights["W{}".format(i)] =\
                self.__weights["W{}".format(i)] - alpha * dW
            self.__weights["b{}".format(i)] =\
                self.__weights["b{}".format(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        trains the deep neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x = []
        y = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                x.append(i)
                y.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            self.gradient_descent(Y, cache, alpha)

        if graph is True:
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename=None):
        """
        saves the instance object to a file in pickle format
        """
        if filename is None:
            return None
        if not filename.lower().endswith('.pkl'):
            filename += '.pkl'
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()

    def load(filename=""):
        """
        loads a pickled DeepNeuralNetwork object
        """
        try:
            file = open(filename, "rb")
            return pickle.load(file)

        except FileNotFoundError:
            return None
