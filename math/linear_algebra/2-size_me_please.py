#!/usr/bin/env python3
"""
Task 2
"""
def matrix_shape(matrix):
    """
    Function that returns the shape of a matrix
    """
    matrix_size = []

    if len(matrix) > 1:
        matrix_size.append(len(matrix))
        if isinstance(matrix[0], list):
            matrix_size.append(len(matrix[0]))
            if isinstance(matrix[0][0], list):
                matrix_size.append(len(matrix[0][0]))
    elif len(matrix) == 1:
        matrix_size.append(len(matrix))
    return matrix_size
