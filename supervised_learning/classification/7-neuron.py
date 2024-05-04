#!/usr/bin/env python3
"""
Task 7
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Class Neuron
    """

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for W
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for b
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for A
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        return -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """
        Evaluates the neuron predictions
        """
        A = self.forward_prop(X)
        return np.round(A).astype(int), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W -= (alpha * dw).T
        self.__b -= alpha * db
        self.__A = self.forward_prop(X)
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        allCosts = []
        allIters = []
        for i in range(iterations + 1):
            self.__A = self.forward_prop(X)
            if i % step == 0 or i == 0 or i == iterations:
                cost = self.cost(Y, self.__A)
                allCosts.append(cost)
                allIters.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            self.__W, self.__b = self.gradient_descent(X, Y, self.__A, alpha)
        if graph is True:
            plt.plot(allIters, allCosts, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
