#!/usr/bin/env python3
"""
Task 0
"""

from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    input_shape = (224, 224, 3)
    inputs = K.Input(shape=input_shape)

    # Initial layers before the inception modules
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(inputs)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception blocks
    x = inception_block(x, (64, 96, 128, 16, 32, 32))
    x = inception_block(x, (128, 128, 192, 32, 96, 64))
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_block(x, (192, 96, 208, 16, 48, 64))
    x = inception_block(x, (160, 112, 224, 24, 64, 64))
    x = inception_block(x, (128, 128, 256, 24, 64, 64))
    x = inception_block(x, (112, 144, 288, 32, 64, 64))
    x = inception_block(x, (256, 160, 320, 32, 128, 128))
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_block(x, (256, 160, 320, 32, 128, 128))
    x = inception_block(x, (384, 192, 384, 48, 128, 128))

    # Average pooling, dropout, and dense layers
    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Flatten()(x)
    outputs = K.layers.Dense(1000, activation='softmax')(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=outputs)

    return model
