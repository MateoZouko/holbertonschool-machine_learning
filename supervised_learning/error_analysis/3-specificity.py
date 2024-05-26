#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each
    class in a confusion matrix
    """

    true_positive = np.diag(confusion)
    false_positive = np.sum(confusion, axis=0) - true_positive
    false_negative = np.sum(confusion, axis=1) - true_positive
    true_negative = np.sum(confusion) - (false_positive + false_negative + true_positive)
    specificity = true_negative / (true_negative + false_positive)

    return specificity