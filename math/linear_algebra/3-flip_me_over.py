#!/usr/bin/env python3
"""
Task 3
"""


def matrix_transpose(matrix):
    """
    Function that returns the transpose of a 2D matrix
    """
    return (list(map(list, zip(*matrix))))
