#!/usr/bin/env python3
"""
Task 1
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    This function creates a sparse autoencoder model.
    """
    input_layer = keras.Input(shape=(input_dims,))

    encoded = input_layer
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    latent_layer = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)

    decoded = latent_layer
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(decoded)

    encoder = keras.Model(inputs=input_layer, outputs=latent_layer)

    latent_input = keras.Input(shape=(latent_dims,))
    reconstructed = latent_input
    for nodes in reversed(hidden_layers):
        reconstructed = keras.layers.Dense(nodes,
                                           activation='relu')(reconstructed)
    reconstructed = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(reconstructed)
    decoder = keras.Model(inputs=latent_input, outputs=reconstructed)

    auto_input = keras.Input(shape=(input_dims,))
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(inputs=auto_input, outputs=decoded_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
