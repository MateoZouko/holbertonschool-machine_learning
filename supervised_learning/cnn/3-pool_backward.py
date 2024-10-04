#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs back propagation over a pooling layer of a
    neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh + 1)
    w_new = int((w_prev - kw) / sw + 1)

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for len in range(c_prev):
                    if mode == 'max':
                        A = A_prev[i, j * sh:j * sh +
                                   kh, k * sw:k * sw + kw, len]
                        mask = (A == np.max(A))
                        dA_prev[i, j * sh:j * sh + kh, k
                                * sw:k * sw + kw, len] += \
                            np.multiply(mask, dA[i, j, k, len])
                    elif mode == 'avg':
                        da = dA[i, j, k, len]
                        average = da / (kh * kw)
                        Z = np.ones(kernel_shape) * average
                        dA_prev[i, j * sh:j * sh + kh,
                                k * sw:k * sw + kw, len] += Z

    return dA_prev
