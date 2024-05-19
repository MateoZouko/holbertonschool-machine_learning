#!/usr/bin/env python3
"""
Task 11
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay in numpy
    Parametros:
    alpha: the original learning rate
    decay_rate: the weight used to determine the rate at which alpha will decay
    global_step: the number of passes of gradient descent that have elapsed
    decay_step: the number of passes of gradient descent that should
                occur before alpha is decayed further
    """

    alpha_act = alpha / (1 + decay_rate * np.floor(global_step / decay_step))

    return alpha_act
