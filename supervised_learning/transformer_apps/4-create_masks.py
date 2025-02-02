#!/usr/bin/env python3
"""
Task 4
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for the input sequence.
    The mask is a tensor of 0s and 1s, where 1 indicates a padding token.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for the target sequence.
    The mask prevents the decoder from attending to future tokens.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    dec_target_padding_mask = create_padding_mask(target)

    combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
