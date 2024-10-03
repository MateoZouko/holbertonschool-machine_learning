#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    if kh % 2 == 0:
        ph = kh // 2
    if kw % 2 == 0:
        pw = kw // 2
    output = np.zeros((m, h, w))
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(h):
        for j in range(w):
            output[:, i, j] = (images[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))
    return output
