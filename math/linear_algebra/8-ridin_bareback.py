#!/usr/bin/env python3
"""
Task 8
"""


def mat_mul(mat1, mat2):
    """
    Function that performs matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat2)):
                sum += mat1[i][k] * mat2[k][j]
            row.append(sum)
        result.append(row)
    return result
