#!/usr/bin/env python3
"""
Task 0
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    # Path 1: 1x1 convolution
    conv1 = Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    # Path 2: 1x1 convolution followed by 3x3 convolution
    conv3r = Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv3 = Conv2D(F3, (3, 3), padding='same', activation='relu')(conv3r)

    # Path 3: 1x1 convolution followed by 5x5 convolution
    conv5r = Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv5 = Conv2D(F5, (5, 5), padding='same', activation='relu')(conv5r)

    # Path 4: Max pooling followed by 1x1 convolution
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    convmax = Conv2D(FPP, (1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate all the paths
    output = concatenate([conv1, conv3, conv5, convmax], axis=-1)

    return output
