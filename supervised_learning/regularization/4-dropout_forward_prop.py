#!/usr/bin/env python3
"""
Task 4
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Forward Propagation with Dropout
    """

    cache = dict()
    for i in range(L):
        if i == 0:
            cache['A0'] = X
        else:
            z = np.dot(weights[f"W{i}"], cache[f"A{i-1}"]
                       ) + weights[f"b{i}"]

            tanh_z = np.tanh(z)
            cache[f"A{i}"] = tanh_z

            cache[f"D{i}"] = np.random.rand(
                cache[f"A{i}"].shape[0],
                cache[f"A{i}"].shape[1]) < keep_prob

            cache[f"D{i}"] = cache[f"D{i}"].astype(int)
            cache[f"A{i}"] *= cache[f"D{i}"]
            cache[f"A{i}"] /= keep_prob

    cache[f"A{L}"] = np.dot(weights[f"W{L}"],
                            cache[f"A{L-1}"]) + weights[f"b{L}"]
    cache[f"A{L}"] = np.exp(cache[f"A{L}"]
                            )/np.sum(np.exp(cache[f"A{L}"]), axis=0)
    return cache
