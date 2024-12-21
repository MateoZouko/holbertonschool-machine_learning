#!/usr/bin/env python3
"""
Task 3
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    This function creates a variational autoencoder.
    """
    inputs = keras.Input(shape=(input_dims,))

    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dims,))

    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    outputs = decoder(encoder(inputs)[0])

    auto = keras.Model(inputs, outputs, name="autoencoder")

    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(
            z_mean) - keras.backend.exp(z_log_var), axis=-1)
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)

    auto.compile(optimizer=keras.optimizers.Adam())

    return encoder, decoder, auto
