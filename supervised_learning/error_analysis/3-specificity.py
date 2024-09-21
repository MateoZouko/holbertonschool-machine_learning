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
    classes = confusion.shape[0]
    specificity_scores = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(confusion) - (np.sum
                                              (confusion[i, :]) + np.sum(
                                                 confusion[:, i]) -
                                              confusion[i, i])

        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        specificity_scores[i] = true_negatives / (true_negatives
                                                  + false_positives)

    return specificity_scores
