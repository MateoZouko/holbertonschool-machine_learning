#!/usr/bin/env python3
"""
Task 9
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    This function finds the best number of clusters for a GMM using the
    Bayesian Information Criterion.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int)
                             or kmax <= 0 or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if (pi is None or m is None or S is None or g is None
                or log_likelihood is None):
            return None, None, None, None

        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)

        bic = p * np.log(n) - 2 * log_likelihood

        log_likelihoods.append(log_likelihood)
        bics.append(bic)

        if k == kmin or bic < best_bic:
            best_k = k
            best_result = (pi, m, S)
            best_bic = bic

    log_likelihoods = np.array(log_likelihoods)
    bics = np.array(bics)

    return best_k, best_result, log_likelihoods, bics
