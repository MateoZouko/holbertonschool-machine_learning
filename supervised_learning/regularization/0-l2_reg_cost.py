#!/usr/bin/env python3
"""
Task 0
"""

import tensorflow as tf


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    function
    """
    l2 = 0
    for i in range(1, L + 1):
        l2 += tf.nn.l2_loss(weights['W' + str(i)])
    return cost + (lambtha / (2 * m)) * l2
