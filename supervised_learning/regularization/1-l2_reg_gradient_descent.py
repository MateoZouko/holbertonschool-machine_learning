#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Calculates one pass of gradient descent on a DNN"""
    m = Y.shape[1]

    A_prev = cache["A" + str(L)]
    dZ = A_prev - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)] if i > 1 else cache["A0"]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dZ = np.dot(W.T, dZ) * (1 - A_prev ** 2)

        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = b - alpha * db
