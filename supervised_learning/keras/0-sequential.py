#!/usr/bin/env python3
"""
Task 0
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Task 0
    """
    model = K.Sequential()
    model.add(
        K.layers.Dense(
            layers[0], input_shape=(nx,),
            activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )
    )
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(
            K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            )
        )
    return model
