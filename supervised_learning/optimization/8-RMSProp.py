#!/usr/bin/env python3
"""
Task 8
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    sets up the RMSProp optimization algorithm in TensorFlow
    Parámetros:
    alpha -- tasa de aprendizaje (learning rate)
    beta2 -- factor de descuento RMSProp (rho en TensorFlow)
    epsilon -- número pequeño para evitar la división por cero
    """

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2, epsilon=epsilon)
    return optimizer
