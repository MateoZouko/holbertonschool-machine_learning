#!/usr/bin/env python3
"""
Task 13
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    """
    init = tf.keras.layers.Dense(n, 
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'))(prev)
    batchN = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones', )(init)
    activation = tf.keras.layers.Activation(activation)(batchN)
    return activation
