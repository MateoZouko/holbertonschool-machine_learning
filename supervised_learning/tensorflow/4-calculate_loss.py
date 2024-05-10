#!/usr/bin/env python3
"""
Task 3
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross-entropy loss of a prediction
    Arguments:
     - y is a placeholder for the labels of the input data
     - y_pred is a tensor containing the network’s predictions
    Returns:
     The loss of the prediction
    """

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
