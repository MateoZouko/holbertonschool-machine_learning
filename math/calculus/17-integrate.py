#!/usr/bin/env python3
"""
Task 17
"""


def poly_integral(poly, C=0):
    """
    poly integral
    """
    if type(C) not in (int, float):
        return None
    if type(poly) is not list or poly == []:
        return None
    for i in poly:
        if type(i) not in (int, float):
            return None

    poly_int = [C]
    for i in range(len(poly)):
        if poly[i]/(i + 1) % 1 <= 0.0:
            poly_int.append(int(poly[i]/(i + 1)))
        elif isinstance(poly[i]/(i + 1), float):
            poly_int.append(float(poly[i]/(i + 1)))

    while poly_int[-1] == 0 and len(poly_int) > 1:
        poly_int.pop(-1)
    return poly_int
