#!/usr/bin/env python3
"""
Task 8
"""

import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork
    """

    def __init__(self, nx, nodes):
        """
        Class constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        getter function for W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        getter function for b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        getter function for A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        getter function for W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        getter function for b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        getter function for b2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        return -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.round(A2).astype(int)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dz1 = A1 - Y
        dz2 = A2 - Y
        dw1 = np.matmul(X, dz1.T) / m
        dw2 = np.matmul(X, dz2.T) / m
        db1 = np.sum(dz1) / m
        db2 = np.sum(dz2) / m
        self.__W1 -= (alpha * dw1).T
        self.__b1 -= alpha * db1
        self.__A1 = self.forward_prop(X)
        self.__W2 -= (alpha * dw2)
        self.__b2 -= alpha * db2
        self.__A2 = self.forward_prop(X)
        return self.__W1, self.__b1, self.__W2, self.__b2
