#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    """
    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes))

    for i in range(len(labels)):
        true_lable = np.argmax(labels[i])
        predicted_lable = np.argmax(logits[i])

        confusion_matrix[true_lable, predicted_lable] += 1

    return confusion_matrix
