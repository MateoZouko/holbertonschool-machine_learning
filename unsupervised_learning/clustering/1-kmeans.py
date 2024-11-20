#!/usr/bin/env python3
"""
Task 1:
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Parameters:
        X [numpy.ndarray of shape (n, d)]:
            Dataset for K-means clustering
        k [int]:
            Number of clusters
        iterations [int]:
            Maximum number of iterations

    Returns:
        C, clss:
            C [numpy.ndarray of shape (k, d)]:
                Centroid means for each cluster
            clss [numpy.ndarray of shape (n,)]:
                Index of the cluster each data point belongs to
        None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize cluster centroids using a multivariate uniform distribution
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Compute distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([
            X[clss == i].mean(axis=0) if np.any(clss == i)
            else np.random.uniform(low, high, size=(d,))
            for i in range(k)
        ])

        # Check for convergence (no change in centroids)
        if np.all(C == new_C):
            break

        C = new_C

    return C, clss
