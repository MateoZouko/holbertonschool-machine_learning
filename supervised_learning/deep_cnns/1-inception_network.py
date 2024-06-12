#!/usr/bin/env python3
"""
Task 1
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    To be defined
    """
    input_1 = K.Input(
        shape=[224, 224, 3],
        )

    conv2d = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation="ReLU"
        )(input_1)

    max_pooling2d = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
        )(conv2d)

    conv2d_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation="ReLU"
        )(max_pooling2d)

    conv2d_2 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation="ReLU"
        )(conv2d_1)

    max_pooling2d_1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
        )(conv2d_2)

    # inception 3a
    out3a = inception_block(max_pooling2d_1, [64, 96, 128, 16, 32, 32])
    # inception 3b
    out3b = inception_block(out3a, [128, 128, 192, 32, 96, 64])

    max_pooling2d_4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
        )(out3b)

    # inception 4a
    out4a = inception_block(max_pooling2d_4, [192, 96, 208, 16, 48, 64])
    # inception 4b
    out4b = inception_block(out4a, [160, 112, 224, 24, 64, 64])
    # inception 4c
    out4c = inception_block(out4b, [128, 128, 256, 24, 64, 64])
    # inception 4e
    out4d = inception_block(out4c, [112, 144, 288, 32, 64, 64])
    # inception 4e
    out4e = inception_block(out4d, [256, 160, 320, 32, 128, 128])

    max_pooling2d_10 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
        )(out4e)

    # inception 5a
    out5a = inception_block(max_pooling2d_10, [256, 160, 320, 32, 128, 128])
    # inception 5b
    out5b = inception_block(out5a, [384, 192, 384, 48, 128, 128])

    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding="valid"
        )(out5b)

    dropout = K.layers.Dropout(rate=0.40)(average_pooling2d)

    dense = K.layers.Dense(
        units=1000,
        activation="softmax",
        )(dropout)

    model = K.Model(inputs=input_1, outputs=dense)

    return model
