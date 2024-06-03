#!/usr/bin/env python3
"""
Model Prediction
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network
    """
    predictions = network.predict(data, verbose=verbose)
    return predictions
