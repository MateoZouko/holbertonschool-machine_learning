#!/usr/bin/env python3
"""
Task 1
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    function that creates a layer
    """
    w = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.keras.layers.Dense(n, activation, name='layer',
                            kernel_initializer=w)
    return layer(prev)
