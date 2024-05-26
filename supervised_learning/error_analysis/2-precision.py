#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def precision(confusion):
    """
    calculates the precision for each
    class in a confusion matrix
    """

    diagonal = np.diag(confusion)
    column_sum = np.sum(confusion, axis=0)
    precision = diagonal / column_sum

    return precision
