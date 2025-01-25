#!/usr/bin/env python3
"""
Task 2
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    This class decodes input sequences for machine translation using
    a GRU-based RNN and an attention mechanism.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        This method initialize the RNNDecoder.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")

        self.F = tf.keras.layers.Dense(units=vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        This method performs the forward pass to decode a single word.
        """
        context, _ = self.attention(s_prev, hidden_states)

        context_exp = tf.expand_dims(context, axis=1)

        x_emb = self.embedding(x)
        concat_input = tf.concat([context_exp, x_emb], axis=-1)
        output, s = self.gru(concat_input)

        y = self.F(output)
        y = tf.squeeze(y, axis=1)

        return y, s
