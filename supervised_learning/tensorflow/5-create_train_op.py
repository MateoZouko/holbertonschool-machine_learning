#!/usr/bin/env python3
"""
Task 5
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network
    """
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train_op
