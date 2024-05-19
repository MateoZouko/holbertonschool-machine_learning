#!/usr/bin/env python3
"""
Task 7
"""

import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    """
    optimizer = var.tf.keras.optimizers.RMSprop(alpha, beta2, epsilon, grad, s)

    return optimizer
