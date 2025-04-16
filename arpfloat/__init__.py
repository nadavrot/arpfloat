#!/usr/bin/env python3
"""
ARPFloat: Arbitrary Precision Floating-Point Library

This library provides arbitrary precision floating-point arithmetic with
configurable precision and rounding modes. It implements IEEE 754
semantics and supports standard arithmetic operations.

Examples:
    >>> from arpfloat import Float, FP16
    >>> x = Float.from_f64(2.5).cast(FP16)
    >>> y = Float.from_f64(1.5).cast(FP16)
    >>> z = x + y  # Result is 4.0
    >>> print(z)
    4

Constants:
    BF16, FP16, FP32, FP64, FP128, FP256: Standard floating-point formats
    pi, e, ln2, zero: Mathematical constants
"""

from ._arpfloat import PyFloat as Float
from ._arpfloat import PySemantics as Semantics
from ._arpfloat import pi, e, ln2, zero, from_i64


# Define standard floating-point types
# Parameters match IEEE 754 standard formats
BF16 = Semantics(8, 7, "NearestTiesToEven")  # BFloat16
FP16 = Semantics(5, 11, "NearestTiesToEven")  # Half precision
FP32 = Semantics(8, 24, "NearestTiesToEven")  # Single precision
FP64 = Semantics(11, 53, "NearestTiesToEven")  # Double precision
FP128 = Semantics(15, 113, "NearestTiesToEven")  # Quadruple precision
FP256 = Semantics(19, 237, "NearestTiesToEven")  # Octuple precision

version = "0.1.10"
