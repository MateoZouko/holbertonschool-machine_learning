#!/usr/bin/env python3

"""
Task 9
"""


def summation_i_squared(n):
    """
    Write a function def summation_i_squared(n):
    that calculates
    """
    if isinstance(n, int) and n >= 1:
        return (n * (n+1) * (2*n+1) // 6)
    else:
        return None
