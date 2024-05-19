#!/usr/bin/env python3
"""
Task 14
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a
    neural network in tensorflow

    Parámetros:
    prev: salida activada de la capa anterior
    n: número de nodos en la capa a ser creada
    activation: función de activación que debe serusada en la salida de la capa
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(units=n,
                                        kernel_initializer=initializer)(prev)

    mean, variance = tf.nn.moments(dense_layer, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    epsilon = 1e-7
    normalized_output = tf.nn.batch_normalization(
        dense_layer, mean, variance, beta, gamma, epsilon
    )

    activated_output = activation(normalized_output)

    return activated_output
