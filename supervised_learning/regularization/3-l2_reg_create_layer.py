#!/usr/bin/env python3
"""
Task 3
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.
    """

    initilizer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha),
        kernel_initializer=initilizer
    )(prev)

    return layer
