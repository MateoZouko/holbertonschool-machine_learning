#!/usr/bin/env python3
"""
Task 9
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    This class creates the encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        This method initializes the Encoder.
        """
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        This method performs the forward pass for the Encoder.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x
