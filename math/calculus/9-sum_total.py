#!/usr/bin/env python3
"""
Task 9
"""


def summation_i_squared(n):
    """
    ssummation i squared
    """

    if n is None:
        return None
    if type(n) is not int:
        return None
    if n <= 0:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
