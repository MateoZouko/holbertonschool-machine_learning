#!/usr/bin/env python3
"""
Task 2
"""


import numpy as np


class Neuron:
    """
    class that represents a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter W
        """
        return (self.__W)

    @property
    def b(self):
        """
        getter b
        """
        return (self.__b)

    @property
    def A(self):
        """
        getter A
        """
        return (self.__A)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)
