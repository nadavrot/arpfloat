#!/usr/bin/env python3
from ._arpfloat import PyFloat as Float
from ._arpfloat import PySemantics as Semantics
from ._arpfloat import pi, e

# Define standard floating-point types
# Parameters match IEEE 754 standard formats
BF16 = Semantics(8, 7, "NearestTiesToEven")  # BFloat16
FP16 = Semantics(5, 11, "NearestTiesToEven")  # Half precision
FP32 = Semantics(8, 24, "NearestTiesToEven")  # Single precision
FP64 = Semantics(11, 53, "NearestTiesToEven")  # Double precision
FP128 = Semantics(15, 113, "NearestTiesToEven")  # Quadruple precision
FP256 = Semantics(19, 237, "NearestTiesToEven")  # Octuple precision
