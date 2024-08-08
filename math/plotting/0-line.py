#!/usr/bin/env python3
"""
Task 0
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    plot y as a line graph
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, color='red')
    plt.xlim(0, 10)
