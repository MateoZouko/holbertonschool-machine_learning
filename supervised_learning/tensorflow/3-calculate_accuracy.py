#!/usr/bin/env python3
"""
Task 3
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction
    """

    return y_pred/y
