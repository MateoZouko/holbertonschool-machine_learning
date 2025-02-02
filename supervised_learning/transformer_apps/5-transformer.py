#!/usr/bin/env python3
"""
Task 5 Transformer
"""

import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.
    """
    pos_encoding_vectors = np.zeros(shape=(max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm // 2):
            div_term = 10000 ** (2 * i / dm)

            pos_encoding_vectors[pos, 2*i] = np.sin(pos / div_term)

            pos_encoding_vectors[pos, 2*i + 1] = np.cos(pos / div_term)

    return pos_encoding_vectors


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
    """
    dk = tf.cast(Q.shape[-1], dtype=tf.float32)

    scores = tf.matmul(Q, K, transpose_b=True)

    scaled_scores = scores / tf.sqrt(dk)

    if mask is not None:
        scaled_scores += (mask * -1e-9)

    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    This class represents the multi head attention mechanism.
    """

    def __init__(self, dm, h):
        """
        Initializes the MultiHeadAttention layer.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the input into multiple heads for multi-head attention.
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Computes the multi-head attention.
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attention_output, weights = sdp_attention(Q, K, V, mask)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention_output, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    This class represents a transformer's encoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the encoder block.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass through the encoder block.
        """
        mha_output, _ = self.mha(x, x, x, mask)
        mha_output = self.dropout1(mha_output, training=training)
        output1 = self.layernorm1(x + mha_output)

        ff_output = self.dense_hidden(output1)
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout2(ff_output, training=training)

        output2 = self.layernorm2(output1 + ff_output)

        return output2


class DecoderBlock(tf.keras.layers.Layer):
    """
    This class represents a transformer's decoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the decoder block.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the transformer's decoder block.
        """
        masked_mha_output, _ = self.mha1(x, x, x, look_ahead_mask)
        masked_mha_output = self.dropout1(masked_mha_output, training=training)
        output1 = self.layernorm1(x + masked_mha_output)

        mha2_output, _ = self.mha2(output1, encoder_output, encoder_output,
                                   padding_mask)
        mha2_output = self.dropout2(mha2_output)

        output2 = self.layernorm2(mha2_output + output1)

        ff_output = self.dense_hidden(output2)
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout3(ff_output, training=training)

        output2 = self.layernorm3(ff_output + output2)

        return output2


class Encoder(tf.keras.layers.Layer):
    """
    This class represents an transformer's Encoder.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Encoder.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass through the `Encoder`.
        """
        input_seq_len = x.shape[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    This class represents an transformer's Decoder.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Decoder.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the `Decoder`.
        """
        input_seq_len = x.shape[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)

        return x


class Transformer(tf.keras.Model):
    """
    This class represents a complete transformer network.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initializes the Transformer model.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Forward pass through the Transformer network.
        """
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
