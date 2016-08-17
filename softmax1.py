#! /usr/bin/env python
""" Softmax """

import numpy as np

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
scores = [1.0, 2.0, 3.0]

def softmax(x):
    """ Compute softmax values for x. """
    ex = np.exp(x)
    s = np.sum(ex, axis=0)
    return np.divide(ex, s)

print (softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
