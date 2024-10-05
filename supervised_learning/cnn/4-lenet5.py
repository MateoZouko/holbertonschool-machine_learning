#!/usr/bin/env python3
"""
Task 4
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(X, Y):
    """
    Function that builds a modified version of the LeNet-5 architecture using
    tensorflow
    """
    init = tf.initializers.variance_scaling()
    activation = tf.nn.relu

    conv1 = tf.layers.conv2d(X, filters=6, kernel_size=(5, 5), padding='same',
                             activation=activation, kernel_initializer=init)

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))

    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(5, 5),
                             padding='valid', activation=activation,
                             kernel_initializer=init)

    pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2))

    flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flatten, units=120, activation=activation,
                          kernel_initializer=init)

    fc2 = tf.layers.dense(fc1, units=84, activation=activation,
                          kernel_initializer=init)

    output = tf.layers.dense(fc2, units=10, kernel_initializer=init)

    loss = tf.losses.softmax_cross_entropy(Y, output)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, accuracy
