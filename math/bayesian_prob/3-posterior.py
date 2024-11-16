#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    Parameters:
    - x: Number of patients that develop severe side effects (int)
    - n: Total number of patients observed (int)
    - P: 1D numpy.ndarray of hypothetical
    probabilities (float values in [0, 1])
    - Pr: 1D numpy.ndarray of prior beliefs about P (float values in [0, 1])

    Returns:
    - A 1D numpy.ndarray containing the posterior probabilities
    """
    # Input validation
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError(
            "All values in Pr must be in the range [0, 1]"
        )
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood: binomial coefficient * P^x * (1 - P)^(n - x)
    fact = (np.math.factorial(n) /
            (np.math.factorial(x) * np.math.factorial(n - x)))
    likelihood = fact * (P ** x) * ((1 - P) ** (n - x))

    # Calculate the marginal probability (evidence)
    marginal_probability = np.sum(likelihood * Pr)

    # Calculate the posterior probability
    posterior_probability = (likelihood * Pr) / marginal_probability

    return posterior_probability
