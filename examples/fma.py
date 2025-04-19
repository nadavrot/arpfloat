import numpy as np
from arpfloat import FP32, fp64, Semantics, zero, fma

# Create two random numpy arrays in the range [0,1)
A0 = np.random.rand(1024)
A1 = np.random.rand(1024)

# Create the fp8 format (4 exponent bits, 3 mantissa bits + 1 implicit bit)
FP8 = Semantics(4, 3 + 1, "NearestTiesToEven")

# Convert the arrays to FP8
B0 = [fp64(x).cast(FP8) for x in A0]
B1 = [fp64(x).cast(FP8) for x in A1]

acc = zero(FP32)
for x, y in zip(B0, B1):
    acc = fma(x.cast(FP32), y.cast(FP32), acc)

print("Using fp8/fp32 arithmetic: ", acc)
print("Using fp32 arithmetic    : ", np.dot(A0, A1))