#!/usr/bin/env python3
"""
Task 0
"""
import numpy as np


class Neuron:
    """
    class Neuron that defines a single
    neuron performing binary classification
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
