#!/usr/bin/env python3
"""
Task 9
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    function that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    """

    op = tf.keras.optimizers.Adam(alpha, beta1, beta2, epsilon)

    return op
