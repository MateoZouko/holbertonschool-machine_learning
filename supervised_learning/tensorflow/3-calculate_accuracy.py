#!/usr/bin/env python3
"""
Task 3
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction
    """
    equality = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
