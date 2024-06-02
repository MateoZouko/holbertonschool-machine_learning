#!/usr/bin/env python3
"""
Task 3
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False, validation_data=None):
    """
    Trains a model using mini-batch gradient descent
    """

    history = network.fit(
        data, labels, batch_size=batch_size, epochs=epochs,
        validation_data=validation_data,
        verbose=verbose, shuffle=shuffle)
    return history
