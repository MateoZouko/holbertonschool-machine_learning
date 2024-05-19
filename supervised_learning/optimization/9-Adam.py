#!/usr/bin/env python3
"""
Task 9
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using
    the Adam optimization algorithm
    Parámetros:
    alpha -- tasa de aprendizaje
    beta1 -- peso usado para el primer momento
    beta2 -- peso usado para el segundo momento
    epsilon -- pequeño número para evitar la división por cero
    var -- numpy.ndarray que contiene la variable a actualizar
    grad -- numpy.ndarray que contiene el gradiente de la variable
    v -- primer momento previo de la variable
    s -- segundo momento previo de la variable
    t -- paso de tiempo usado para la corrección de sesgo
    """

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
