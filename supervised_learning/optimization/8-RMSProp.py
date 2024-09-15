#!/usr/bin/env python3
"""
Task 8
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    function that creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm
    """

    op = tf.train.RMSPropOptimizer(alpha, beta2, epsilon).minimize(loss)

    return optimizer.minimize(loss)
