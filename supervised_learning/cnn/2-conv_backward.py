#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, activation='relu', padding="same", stride=(1, 1)):
    """
    Function that performs back propagation over a convolutional layer of a
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

    dA_prev = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for len in range(c_new):
                    dA_prev[i, j * sh:j * sh + kh, k * sw:k * sw + kw, :] += \
                        W[:, :, :, len] * dZ[i, j, k, len]
                    dW[:, :, :, len] += A_prev_pad[i, j * sh:j * sh + kh,
                                                   k * sw:k * sw +
                                                   kw, :] * dZ[i, j, k, len]
                    db[:, :, :, len] += dZ[i, j, k, len]

    if activation == 'relu':
        dA_prev = np.where(A_prev_pad <= 0, 0, dA_prev)

    return dA_prev, dW, db
