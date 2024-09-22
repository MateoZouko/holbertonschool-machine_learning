#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    function that calculates the cost
    of a neural network with L2 regularization
    """
    norm = 0
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights['W' + str(i)])
    return cost + lambtha / (2 * m) * norm
