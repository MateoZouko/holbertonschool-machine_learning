#!/usr/bin/env python3
"""
Task 4
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    This function calculates the positional encoding for a transformer.
    """
    PE = np.zeros((max_seq_len, dm))

    for pos in range(max_seq_len):
        for j in range(0, dm, 2):
            angle = pos / 10000 ** (j / dm)
            PE[pos, j] = np.sin(angle)
            PE[pos, j + 1] = np.cos(angle)

    return PE
