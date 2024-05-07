#!/usr/bin/env python3*
"""
task 10
"""


def poly_derivative(poly):
    """
    poly derivativee
    """
    list2 = []
    if type(poly) is not list:
        return None
    for i in poly:
        if type(i) is not int:
            return None
    if len(poly) == 1:
        return [0]
    if len(poly) == 0:
        return None

    for i in range(0, len(poly) - 1):

        list2.append(poly[i+1]*(i+1))
    return list2
