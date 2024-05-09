#!/usr/bin/env python3
"""
Task 1
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Function that creates a layer
    """

    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.layers.dense(inputs=prev,
        units=n, activation=activation, kernel_initializer=init)

    return layer
