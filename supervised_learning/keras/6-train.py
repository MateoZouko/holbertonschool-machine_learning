#!/usr/bin/env python3
"""
Task 5
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient
    descent and analyzes validation data,
    with optional early stopping
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stopping_cb = K.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience)
        callbacks.append(early_stopping_cb)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data, verbose=verbose,
                          shuffle=shuffle, callbacks=callbacks)
    return history
