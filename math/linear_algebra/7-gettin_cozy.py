#!/usr/bin/env python3
"""
Task 7
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    if axis == 0:
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    elif axis == 1:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
