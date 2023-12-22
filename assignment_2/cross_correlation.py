import torch
import numpy as np
import matplotlib as plt

def add_border(orig, border_px=1):
    """
    Applies border around an image
    Args:
      orig (ndarray (a,b,c)) : original image
      border_px (int)        : border size, nonnegative, default value = 1
    Returns
      bordered (ndarray (a + 2*border_px, b + 2*border_px, c)) : bordered image
    """
    a, b, c = orig.shape
    bordered = np.zeros((a + 2*border_px, b + 2*border_px, c))
    bordered[border_px:-border_px, border_px:-border_px, :] = orig
    return bordered

orig = np.ones((1,2,3))
bordered = add_border(orig)
print(orig)
print(orig.shape)
print('\n\n\n')
print(bordered.shape)
print(bordered)

arr = np.array([
    [1, 0.5, 0, -0.5, -1],
    [2, 1, 0, -1, 2],
    [3, 1.5, 0, -1.5, -3],
    [2, 1, 0 -1, -2],
    [1, 0.5, 0, -0.5, -1]
])