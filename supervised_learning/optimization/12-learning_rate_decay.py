#!/usr/bin/env python3
"""
Task 11
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    function that creates a learning rate decay operation in tensorflow using
    inverse time decay
    """

    op = tf.train.inverse_time_decay(alpha, global_step=0,
                                     decay_steps=decay_step,
                                     decay_rate=decay_rate, staircase=True)

    return op
