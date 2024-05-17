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
    max_value = np.max(X)
    min_value = np.min(X)

    return max_value, min_value
