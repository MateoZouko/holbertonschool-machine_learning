#!/usr/bin/env python3
"""
Task 6
"""


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer using Keras
    """
    init = K.initializers.he_normal(seed=0)
    activation = K.activations.relu
    Batch_Norm1 = K.layers.BatchNormalization(axis=3)(X)
    ReLU1 = K.layers.Activation(activation)(Batch_Norm1)
    nb_filters *= compression
    nb_filters = int(nb_filters)
    C1 = K.layers.Conv2D(filters=nb_filters,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=init)(ReLU1)
    AP1 = K.layers.AveragePooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding="valid")(C1)
    return AP1, nb_filters
