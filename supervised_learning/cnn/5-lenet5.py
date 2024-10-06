#!/usr/bin/env python3
"""
Task 5
"""


from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of LeNet-5 architecture using Keras
    """
    weights_initializer = K.initializers.he_normal()
    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=weights_initializer)
    output_1 = C1(X)
    P2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))
    output_2 = P2(output_1)
    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='valid',
                         activation=K.activations.relu,
                         kernel_initializer=weights_initializer)
    output_3 = C3(output_2)
    P4 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))
    output_4 = P4(output_3)
    output_42 = K.layers.Flatten()(output_4)
    F5 = K.layers.Dense(
        120,
        activation=K.activations.relu,
        kernel_initializer=weights_initializer)
    output_5 = F5(output_42)
    F6 = K.layers.Dense(
        84,
        activation=K.activations.relu,
        kernel_initializer=weights_initializer)
    output_6 = F6(output_5)
    F7 = K.layers.Dense(
        10,
        kernel_initializer=weights_initializer)
    output_7 = F7(output_6)
    softmax = K.layers.Softmax()(output_7)
    model = K.Model(inputs=X, outputs=softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
