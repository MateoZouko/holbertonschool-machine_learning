#!/usr/bin/env python3
"""
Task 4
"""


def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    else:
        return list(map(lambda x, y: x + y, arr1, arr2))
