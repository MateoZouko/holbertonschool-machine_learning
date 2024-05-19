#!/usr/bin/env python3
"""
Task 13
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a
    neural network using batch normalization
    """

    mean = np.mean(Z, axis=0, keepdims=True)

    variance = np.var(Z, axis=0, keepdims=True)

    normalized_Z = (Z - mean) / np.sqrt(variance + epsilon)

    normalized_Z = gamma * normalized_Z + beta

    return normalized_Z
