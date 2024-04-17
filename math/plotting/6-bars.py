#!/usr/bin/env python3
"""
Task 6
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Create a bar graph with multiple bars per person.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    labels = ['Farrah', 'Fred', 'Felicia']
    width = 0.5
    plt.bar(labels, fruit[0], width, color='r', label='apples')
    plt.bar(labels, fruit[1], width, color='yellow',
            bottom=fruit[0], label='bananas')
    plt.bar(labels, fruit[2], width, color='#ff8000',
            bottom=fruit[0]+fruit[1], label='oranges')
    plt.bar(labels, fruit[3], width, color='#ffe5b4',
            bottom=fruit[0]+fruit[1]+fruit[2], label='peaches')
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.legend()
