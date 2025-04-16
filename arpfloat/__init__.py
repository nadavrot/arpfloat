#!/usr/bin/env python3
"""
ARPFloat: Arbitrary Precision Floating-Point Library

This library provides arbitrary precision floating-point arithmetic with
configurable precision and rounding modes. It implements IEEE 754
semantics and supports standard arithmetic operations.

Examples:
    >>> from arpfloat import Float, FP16
    >>> x = from_f64(FP32, 2.5).cast(FP16)
    >>> y = from_f64(FP32, 1.5).cast(FP16)
    >>> x + y
    4

    >>> sem = Semantics(10, 10, "Zero")
    >>> sem
    Semantics { exponent: 10, precision: 10, mode: Zero }
    >>> Float(sem, False, 1, 13)
    .0507

    >>> arpfloat.pi(arpfloat.FP32)
    3.1415927
    >>> pi(FP16)
    3.14
    >>> pi(BF16)
    3.15

Constants:
    BF16, FP16, FP32, FP64, FP128, FP256: Standard floating-point formats
    pi, e, ln2, zero: Mathematical constants
    Float, Semantics: Classes for representing floating-point numbers and their semantics
    from_i64, from_f64: Constructors for creating Float objects from integers and floats
"""

from ._arpfloat import PyFloat as Float
from ._arpfloat import PySemantics as Semantics
from ._arpfloat import pi, e, ln2, zero, from_i64, from_f64


# Define standard floating-point types
# Parameters match IEEE 754 standard formats
BF16 = Semantics(8, 7, "NearestTiesToEven")  # BFloat16
FP16 = Semantics(5, 11, "NearestTiesToEven")  # Half precision
FP32 = Semantics(8, 24, "NearestTiesToEven")  # Single precision
FP64 = Semantics(11, 53, "NearestTiesToEven")  # Double precision
FP128 = Semantics(15, 113, "NearestTiesToEven")  # Quadruple precision
FP256 = Semantics(19, 237, "NearestTiesToEven")  # Octuple precision

version = "0.1.10"
