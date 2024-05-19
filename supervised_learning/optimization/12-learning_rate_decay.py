#!/usr/bin/env python3
"""
Task 12
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    creates a learning rate decay operation
    in tensorflow using inverse time decay
    Parametros:
    alpha: original learning rate
    decay_rate: weight used to determine the rate at which alpha will decay
    decay_step: number of passes of gradient descent that
                should occur before alpha is decayed further
    """

    global_step = tf.Variable(0, trainable=False, name='global_step')

    learning_rate = tf.compat.v1.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return learning_rate
