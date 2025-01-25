#!/usr/bin/env python3
"""
Task 3
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    This function converts a gensim Word2Vec model to a Keras Embedding layer.
    """
    # Get the vocabulary size and embedding dimension
    vocab_size = len(model.wv.key_to_index)
    embed_size = model.wv.vector_size

    # Initialize the Keras Embedding layer
    keras_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_size,
        embeddings_initializer=tf.constant_initializer(model.wv.vectors)
    )

    return keras_layer
