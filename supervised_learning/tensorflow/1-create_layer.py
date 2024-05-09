#!/usr/bin/env python3
"""
Task 1
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Function that creates a layer
    """

    init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="FAN_AVG", distribution="truncated_normal")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init,
        name='layer')(prev)

    return layer
