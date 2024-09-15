#!/usr/bin/env python3
"""
Task 13
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer
    for a neural network in TensorFlow.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(units=n,
                                        kernel_initializer=initializer)(prev)

    gamma = tf.Variable(initial_value=tf.ones((n,),
                                              dtype=tf.float32),
                        trainable=True, name='gamma')
    beta = tf.Variable(initial_value=tf.zeros((n,),
                                              dtype=tf.float32),
                       trainable=True, name='beta')

    mean, variance = tf.nn.moments(dense_layer, axes=[0])

    epsilon = 1e-7
    batch_norm = tf.nn.batch_normalization(
        dense_layer, mean, variance, beta, gamma, epsilon
    )

    if activation is not None:
        return activation(batch_norm)
    else:
        return batch_norm
