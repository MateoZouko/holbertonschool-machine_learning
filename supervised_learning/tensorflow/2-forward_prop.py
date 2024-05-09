#!/usr/bin/env python3
"""
Task 1
"""

import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    function forward_prop
    """
    create_layer = __import__('1-create_layer').create_layer
    prediction = x

    for i, activation in zip(layer_sizes, activations):
        prediction = create_layer(prediction, i, activation)

    return prediction
