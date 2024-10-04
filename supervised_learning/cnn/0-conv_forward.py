#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional layer of a
    neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    h_new = int((h_prev + 2 * ph - kh) / sh + 1)
    w_new = int((w_prev + 2 * pw - kw) / sw + 1)

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(A_prev_pad[:,
                                                  i * sh:i * sh + kh,
                                                  j * sw:j * sw + kw] *
                                       W[:, :, :, k], axis=(1, 2, 3))

    Z = Z + b
    A = activation(Z)

    return A
