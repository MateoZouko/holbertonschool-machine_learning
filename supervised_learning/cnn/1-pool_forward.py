#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs forward propagation over a pooling layer of a
    neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh + 1)
    w_new = int((w_prev - kw) / sw + 1)

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            if mode == 'max':
                A[:, i, j, :] = np.max(A_prev[:,
                                              i * sh:i * sh + kh,
                                              j * sw:j * sw + kw, :],
                                       axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(A_prev[:,
                                               i * sh:i * sh + kh,
                                               j * sw:j * sw + kw, :],
                                        axis=(1, 2))

    return A
