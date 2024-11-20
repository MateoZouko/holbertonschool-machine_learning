#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
        iterations: positive integer,
        the maximum number of iterations to perform

    Returns:
        C: numpy.ndarray of shape (k, d) containing the
        centroid means for each cluster
        clss: numpy.ndarray of shape (n,) containing the
        index of the cluster each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids using a multivariate uniform distribution
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))
    prev_C = np.copy(C)
    for _ in range(iterations):
        # Compute distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):
            points_in_cluster = X[clss == i]
            if points_in_cluster.shape[0] == 0:
                # Reinitialize centroid if no points are assigned to it
                C[i] = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
            else:
                C[i] = points_in_cluster.mean(axis=0)

        # Check for convergence (no change in centroids)
        if np.all(C == prev_C):
            break
        prev_C = np.copy(C)

    return C, clss
