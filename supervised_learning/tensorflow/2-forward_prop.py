#!/usr/bin/env python3
"""
Task 2
"""

import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    function that creates the forward propagation graph for the neural network
    """
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i == 0:
            y_pred = create_layer(x, layer_sizes[i], activations[i])
        else:
            y_pred = create_layer(y_pred, layer_sizes[i], activations[i])
    return y_pred
