#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, activation='relu',
                  padding="same", stride=(1, 1)):
    """
    Function that performs back propagation
    over a convolutional layer of a
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

    dA_prev = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros((1, 1, 1, c_new))

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for len in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev_pad[i, vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    dA_prev[i, vert_start:vert_end,
                            horiz_start:horiz_end, :] += \
                        W[:, :, :, ] * dZ[i, j, k, len]
                    dW[:, :, :, len] += A_slice * dZ[i, j, k, len]
                    db[:, :, :, len] += dZ[i, j, k, len]

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    if activation == 'relu':
        dA_prev = np.where(A_prev_pad > 0, dA_prev, 0)

    return dA_prev, dW, db
