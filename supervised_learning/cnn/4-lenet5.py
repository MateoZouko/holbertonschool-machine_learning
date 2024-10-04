#!/usr/bin/env python3
"""
Task 4
"""

import numpy as np
import tensorflow as tf


def lenet5(X, Y):
    """
    Function that builds a modified version of the LeNet-5 architecture using
    tensorflow
    """

    tf.keras.backend.clear_session()

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                                     activation='relu',
                                     input_shape=X.shape[1:]))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(120, activation='relu'))

    model.add(tf.keras.layers.Dense(84, activation='relu'))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X, Y, batch_size=32, epochs=10,
                        verbose=1, validation_data=(X, Y))

    return model, history
