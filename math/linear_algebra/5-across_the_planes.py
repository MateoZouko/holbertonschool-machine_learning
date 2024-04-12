#!/usr/bin/env python3
"""
Task 5
"""


def add_matrices2D(mat1, mat2):
    """
    Function that adds two matrices element-wise
    """
    new_mat = []
    
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        for row1, row2 in zip(mat1, mat2):
            new_row = [x + y for x, y in zip(row1, row2)]
            new_mat.append(new_row)
        
    return new_mat
