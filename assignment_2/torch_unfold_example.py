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

# Check for understanding by using a 5x5 kernel

orig = np.arange(36).reshape(1, 1, 6, 6)
orig = torch.Tensor(orig)

unfold_op_5 = torch.nn.Unfold(kernel_size=(5, 5), dilation=1, padding=0, stride=1)

result = unfold_op_5(orig)

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
