#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def precision(confusion):
    """
    Function that calculates the precision
    for each class in a confusion matrix
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
