#!/usr/bin/env python3
"""
Task 4
"""

import numpy as np
import tensorflow.compat.v1 as tf


def lenet5(X, Y):
    """
    Function that builds a modified version of the LeNet-5 architecture using
    tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                             padding='same', activation=activation,
                             kernel_initializer=init)(X)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding='valid', activation=activation,
                             kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(fc1)
    output = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)

    loss = tf.losses.softmax_cross_entropy(Y, output)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.nn.softmax(output)
    accuracy = tf.metrics.accuracy(Y, y_pred)

    return y_pred, train_op, loss, accuracy
