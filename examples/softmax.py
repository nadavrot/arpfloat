import numpy as np
from arpfloat import FP32, BF16, fp64, zero

dtype = BF16

A0 = np.random.rand(6) # Random array in the range [0,1)
B0 = [fp64(x).cast(dtype) for x in A0] # Convert to the emulated format.

# Find the max value.
max_val = max(B0)

# calculate exp(x-max) for each value.
shifted_exp = [(x - max_val).exp() for x in B0]
exp_sum = sum(shifted_exp)

# calculate the softmax: [exp(x-max) / sum(exp(x-max))]
result = [x / exp_sum for x in shifted_exp]
print("Calculated = ", result)

# NumPy's softmax.
np_softmax  = np.exp(A0 - np.max(A0)) / np.exp(A0 - np.max(A0)).sum()
print("Reference = ", np_softmax)


