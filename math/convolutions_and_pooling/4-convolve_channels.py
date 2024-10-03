#!/usr/bin/env python3
"""
Task 4
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        ph, pw = 0, 0
    output_h = int((h - kh + 2 * ph) / sh) + 1
    output_w = int((w - kw + 2 * pw) / sw) + 1
    output = np.zeros((m, output_h, output_w))
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images[:, i * sh: i * sh + kh,
                                      j * sw: j * sw + kw] * kernel
                               ).sum(axis=(1, 2, 3))
    return output
