#!/usr/bin/env python3
"""
Task 0
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Create placeholders
    """

    x = tf.placeholber(tf.float32, shape=[None, nx])
    y = tf.placeholber(tf.float32, shape=[None, classes])
    return x, y
