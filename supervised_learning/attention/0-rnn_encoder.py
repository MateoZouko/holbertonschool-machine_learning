#!/usr/bin/env python3
"""
Task 0
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    This class represents an RNN Encoder for Machine Translation.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        This method initialize the RNNEncoder.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """
        This method initialize the hidden states for the RNN cell to a tensor
        of zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        This method forward pass through the encoder.
        """
        x = self.embedding(x)

        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
