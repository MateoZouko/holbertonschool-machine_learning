#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each
    class in a confusion matrix
    """

    diagonal = np.diag(confusion)
    row_sum = np.sum(confusion, axis=1)
    sensitivity = diagonal / row_sum

    return sensitivity
