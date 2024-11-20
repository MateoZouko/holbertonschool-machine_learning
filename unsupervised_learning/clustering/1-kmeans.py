#!/usr/bin/env python3
"""
task 1
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            contains the number of clusters
        iterations [positive int]:
            contains the maximum number of iterations that should be performed

    if no change in the cluster centroids occurs between iterations,
        the function should return

    initialize the cluster centroids using a multivariate unitform distribution

    if a cluster contains no data points during the update step,
        its centroid should be reinitialized

    should use:
        numpy.random.uniform exactly twice
        at most 2 loops

    returns:
        C, clss:
            C [numpy.ndarray of shape (k, d)]:
                containing the centroid means for each cluster
            clss [numpy.ndarray of shape (n,)]:
                containting the index of the cluster in c
                    that each data point belongs to
        or None, None on failure
    """
    if type(k) is not int or k <= 0:
        return None, None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    centroids = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                                  size=(k, d))
    for i in range(iterations):
        copy = centroids.copy()
        D = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = (X[clss == j]).mean(axis=0)
        D = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        if np.all(copy == centroids):
            return centroids, clss

    return centroids, clss
