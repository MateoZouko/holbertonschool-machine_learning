#!/usr/bin/env python3
"""
Task 16
"""

import numpy as np


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
