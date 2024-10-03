#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on
    grayscale images with custom padding
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h - kh + 1 + 2 * ph
    output_w = w - kw + 1 + 2 * pw
    output = np.zeros((m, output_h, output_w))
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))
    return output
