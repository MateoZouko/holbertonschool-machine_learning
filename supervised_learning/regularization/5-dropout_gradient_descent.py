#!/usr/bin/env python3
"""
Task 5
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Calculates one pass of gradient descent on a DNN
    with dropout regularization
    """
    m = Y.shape[1]

    A_prev = cache["A" + str(L)]

    dZ = A_prev - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)] if i > 1 else cache["A0"]

        cache[f"D{i}"] = np.random.rand(
            A_prev.shape[0],
            A_prev.shape[1]) < keep_prob
        cache[f"D{i}"] = cache[f"D{i}"].astype(int)

        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.dot(W.T, dZ)
            dA_prev *= cache[f"D{i-1}"]
            dA_prev /= keep_prob
            dZ = (1 - A_prev ** 2) * dA_prev

        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = b - alpha * db
