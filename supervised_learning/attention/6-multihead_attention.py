#!/usr/bin/env python3
"""
Task 6
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    This class represents the Multi-Head Attention mechanism for transformers.
    """
    def __init__(self, dm, h):
        """
        This method initializes the MultiHeadAttention layer.
        """
        super(MultiHeadAttention, self).__init__()
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        This method splits the last dimension into (h, depth) and transposes to
        shape (batch, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        This method computes the multi-head attention.
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attention_output, attention_weights = sdp_attention(Q, K, V, mask)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention_output, (batch_size, -1, self.dm)
        )
        output = self.linear(concat_attention)

        return output, attention_weights
