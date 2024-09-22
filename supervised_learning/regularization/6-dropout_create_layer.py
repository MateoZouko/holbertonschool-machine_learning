#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Create a layer of a neural network using dropout.
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    dense_layer = tf.keras.layers.Dense(
        units=n, activation=activation,
        kernel_initializer=initializer)(prev)

    dropout_layer = tf.keras.layers.\
        Dropout(1 - keep_prob)(dense_layer, training=training)

    return dropout_layer
