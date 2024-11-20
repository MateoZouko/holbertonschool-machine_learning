#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Function that performs K-means on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    centroids = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))
    return centroids, None
