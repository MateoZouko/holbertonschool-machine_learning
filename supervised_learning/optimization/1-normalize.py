#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def normalize(X, m, s):
    """
    function that normalizes (standardizes) a matrix
    """
    return (X - m) / s
