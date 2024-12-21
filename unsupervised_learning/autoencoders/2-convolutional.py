#!/usr/bin/env python3
"""
Task 2
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    This function creates a convolutional autoencoder.
    """
    input_img = keras.layers.Input(shape=input_dims)

    x = input_img
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    latent_space = x

    latent_input = keras.layers.Input(shape=latent_dims)

    for idx, units in enumerate(reversed(filters)):
        if idx != len(filters) - 1:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            )
            if idx == 0:
                outputs = layer(latent_input)
            else:
                outputs = layer(outputs)
        else:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )
            outputs = layer(outputs)

        layer = keras.layers.UpSampling2D(size=(2, 2))

        outputs = layer(outputs)

    layer = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )

    outputs = layer(outputs)

    encoder = keras.models.Model(inputs=input_img, outputs=latent_space)

    decoder = keras.models.Model(inputs=latent_input, outputs=outputs)

    auto = keras.models.Model(inputs=input_img,
                              outputs=decoder(encoder(input_img)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
