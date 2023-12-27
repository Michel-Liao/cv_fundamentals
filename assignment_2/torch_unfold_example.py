# Example written by Erich Liang
# Comments written by Michel Liao

import torch
import numpy as np
import matplotlib.pyplot as plt

orig = np.arange(16).reshape(1, 1, 4, 4)
print(type(orig))

orig = torch.Tensor(orig)
print(type(orig))

unfold_op = torch.nn.Unfold(kernel_size=(3, 3), dilation=1, padding=0, stride=1)
# Stride of 1 is the basic step. Move your kernel over by 1 pixel at a time.
result = unfold_op(orig)

print("orig:")
print(orig)
print(orig.shape)

print("result:")
print(result.shape)
# Note result[0, :, i] gives the ith kernel of orig
print(result[0, :, 0])
print(result[0, :, 1])
print(result[0, :, 2])
print(result[0, :, 3])
breakpoint()
