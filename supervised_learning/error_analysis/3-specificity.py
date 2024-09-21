#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity
    for each class in a confusion matrix
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
