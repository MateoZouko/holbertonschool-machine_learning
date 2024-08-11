#!/usr/bin/env python3

"""
Task 10
"""


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial
    """
    if not isinstance(poly, list) or not poly or len(poly) == 0:
        return None
    new_list = []
    for i in range(1, len(poly)):
        new_list.append(poly[i] * i)

    if len(new_list) == 0:
        return [0]
    return new_list
