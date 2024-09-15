#!/usr/bin/env python3
"""
Task 8
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    function that creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm
    """

    op = tf.keras.optimizers.RMSprop(alpha, beta2, epsilon)

    return op
