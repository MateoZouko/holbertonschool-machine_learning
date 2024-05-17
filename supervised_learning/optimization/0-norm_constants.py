#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization
    (standardization) constants of a matrix
    """
    return np.max(X), np.min(X)
