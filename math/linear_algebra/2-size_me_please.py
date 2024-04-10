#!/usr/bin/env python3
"""
Task 2
"""


def matrix_shape(matrix):
    """
    Function that returns the shape of a matrix
    """
    if not matrix:
        return []
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return [len(matrix)]
