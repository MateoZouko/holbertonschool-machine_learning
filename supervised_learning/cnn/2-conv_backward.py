#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a
    convolutional layer of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    h_new = dZ.shape[1]
    w_new = dZ.shape[2]

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph),
                                 (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph),
                                   (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):

                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev_pad[i, vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    dA_prev_pad[i, vert_start:vert_end,
                                horiz_start:horiz_end, :]\
                        += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_slice * dZ[i, h, w, c]

    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
