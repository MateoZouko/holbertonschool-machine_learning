#!/usr/bin/env python3
"""
Task 7
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    """

    if (opt_cost - cost) > threshold:
        return (False, 0)

    elif (opt_cost - cost) <= threshold:
        if count < patience - 1:
            return (False, count + 1)
        if count == patience - 1:
            return (True, patience)
