#!/usr/bin/env python3
"""
Task 4
"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a dataset
    """

    moving_avg = []
    vt = 0

    for t in range(len(data)):
        vt = beta * vt + (1 - beta) * data[t]
        bias_corrected = 1 - beta ** (t + 1)
        corrected_vt = vt / bias_corrected
        moving_avg.append(corrected_vt)
    return moving_avg
