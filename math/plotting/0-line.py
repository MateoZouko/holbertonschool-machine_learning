#!/usr/bin/env python3
"""
Task 0
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot y
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    for x in range (0, 11):
        plt.plot([x, x], [0, y[x]], color='red')
