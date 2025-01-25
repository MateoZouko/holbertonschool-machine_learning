#!/usr/bin/env python3
"""
Task 5
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    This function computes the scaled dot product attention.
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_scores = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_scores += (mask * -1e9)

    weights = tf.nn.softmax(scaled_scores, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
