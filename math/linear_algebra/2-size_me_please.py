#!/usr/bin/env python3
def matrix_shape(matrix):
    matrix_size = []

    if len(matrix) > 1:
        matrix_size.append(int(len(matrix)))
        matrix_size.append(int(len(matrix[0])))
        if isinstance(matrix[0][0], list):
            matrix_size.append(int(len(matrix[0][0])))
    return matrix_size
