#!/usr/bin/env python3
"""
Task 7
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient
    descent and analyzes validation data,
    with optional early stopping,
    learning rate decay, and saving the best model
    """

    callbacks = []

    if early_stopping and validation_data is not None:
        early_stopping_cb = K.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience)
        callbacks.append(early_stopping_cb)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_cb = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay_cb)

    if save_best and filepath is not None and validation_data is not None:
        checkpoint_cb = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                    monitor='val_loss',
                                                    save_best_only=True)
        callbacks.append(checkpoint_cb)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data, verbose=verbose,
                          shuffle=shuffle, callbacks=callbacks)

    return history
