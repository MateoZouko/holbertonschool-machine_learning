#!/usr/bin/env python3
"""
Task 13
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network using batch normalization
    """
    m = Z.shape[0]
    mean = np.sum(Z, axis=0) / m
    variance = np.sum((Z - mean) ** 2, axis=0) / m
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
