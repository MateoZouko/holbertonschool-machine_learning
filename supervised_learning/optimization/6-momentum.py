#!/usr/bin/env python3
"""
Task 6
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os


def create_momentum_op(alpha, beta1):
    """
    function that creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum optimization algorithm
    """
    def momentum_op(loss, var_list):
        """
        loss is the loss of the network
        var_list is the list of all variables to be updated
        """
        optimizer = tf.train.MomentumOptimizer(alpha, beta1)
        train = optimizer.minimize(loss)
        return train
    return momentum_op
