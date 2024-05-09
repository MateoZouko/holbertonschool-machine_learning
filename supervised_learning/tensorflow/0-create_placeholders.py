#!/usr/bin/env python3
"""
Task 0
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Create placeholders
    """

    x = tf.placeholber(tf.float32, shape=[None, nx], name = 'x')
    y = tf.placeholber(tf.int32, shape=[None, classes], name='y')
    return x, y
