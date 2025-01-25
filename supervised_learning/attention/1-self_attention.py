#!/usr/bin/env python3
"""
Task 1
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    This class computes the attention scores and context vector
    for machine translation tasks based on Bahdanau et al. (2015).
    """

    def __init__(self, units):
        """
        This method initializes the SelfAttention layer.
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        This method performs the forward pass to compute the attention
        context and weights.
        """
        s_prev_exp = tf.expand_dims(s_prev, axis=1)

        score = self.V(tf.nn.tanh(self.W(s_prev_exp) + self.U(hidden_states)))

        weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
