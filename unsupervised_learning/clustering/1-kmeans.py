#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
        iterations: positive integer,
        the maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
        clss: numpy.ndarray of shape (n,) containing
        the cluster indices for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids using a multivariate uniform distribution
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, d))

    for _ in range(iterations):
        # Compute distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Compute new centroids as the mean of points in each cluster
        new_C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i)
                          else np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0), size=(d,))
                          for i in range(k)])

        # Check for convergence (no change in centroids)
        if np.all(new_C == C):
            break

        C = new_C

    return C, clss
