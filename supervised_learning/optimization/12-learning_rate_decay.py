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

    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True,
        name='InverseTimeDecay'
    )

    return learning_rate
