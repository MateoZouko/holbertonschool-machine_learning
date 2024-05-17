#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalize a matrix
    """

    return ((X - m) / s)
