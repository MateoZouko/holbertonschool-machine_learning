#!/usr/bin/env python3
"""
Task 0
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Function that builds an inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Path 1: 1x1 convolution
    conv1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                            activation='relu')(A_prev)

    # Path 2: 1x1 convolution followed by 3x3 convolution
    conv3r = K.layers.Conv2D(F3R, (1, 1),
                             padding='same', activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3, (3, 3),
                            padding='same', activation='relu')(conv3r)

    # Path 3: 1x1 convolution followed by 5x5 convolution
    conv5r = K.layers.Conv2D(F5R, (1, 1),
                             padding='same', activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                            activation='relu')(conv5r)

    # Path 4: Max pooling followed by 1x1 convolution
    maxpool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                    padding='same')(A_prev)
    convmax = K.layers.Conv2D(FPP, (1, 1), padding='same',
                              activation='relu')(maxpool)

    # Concatenate all the paths
    output = K.layers.concatenate([conv1, conv3, conv5, convmax],
                                  axis=-1)

    return output
