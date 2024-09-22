#!/usr/bin/env python3
"""
Task 1
"""

import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    function that calculates the cost of a neural network with L2 regularization
    """
    return cost + model.losses
