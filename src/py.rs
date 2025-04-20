use crate::{BigInt, Float, RoundingMode, Semantics};
use core::ops::{Add, Div, Mul, Sub};
use pyo3::prelude::*;
use std::format;
use std::string::String;
use std::string::ToString;

/// Semantics class defining precision and rounding behavior.
///
/// This class encapsulates the parameters that define the precision and
/// rounding behavior of floating-point operations.
#[pyclass]
struct PySemantics {
    inner: Semantics,
}

#[pymethods]
impl PySemantics {
    /// Create a new semantics object.
    ///
    /// Args:
    ///     exp_size: The size of the exponent in bits
    ///     mantissa_size: The size of the mantissa, including the implicit bit
    ///     rounding_mode: The rounding mode to use:
    ///         "NearestTiesToEven", "NearestTiesToAway",
    ///         "Zero", "Positive", "Negative"
    #[new]
    fn new(exp_size: i64, mantissa_size: u64, rounding_mode_str: &str) -> Self {
        let rm = RoundingMode::from_string(rounding_mode_str);
        assert!(rm.is_some(), "Invalid rounding mode");
        let sem = Semantics::new(
            exp_size as usize,
            mantissa_size as usize,
            rm.unwrap(),
        );
        PySemantics { inner: sem }
    }
    /// Returns the length of the exponent in bits.
    fn get_exponent_len(&self) -> usize {
        self.inner.get_exponent_len()
    }
    /// Returns the length of the mantissa in bits.
    fn get_mantissa_len(&self) -> usize {
        self.inner.get_mantissa_len()
    }
    /// Returns the rounding mode as a string.
    fn get_rounding_mode(&self) -> String {
        self.inner.get_rounding_mode().as_string().to_string()
    }
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// A class representing arbitrary precision floating-point numbers.
///
/// This class implements IEEE 754-like floating-point arithmetic with
///  configurable precision and rounding modes.
#[pyclass]
struct PyFloat {
    inner: Float,
}

#[pymethods]
impl PyFloat {
    /// Create a new floating-point number.
    ///
    /// Args:
    ///     sem: The semantics (precision and rounding mode) for this number
    ///     is_negative: Whether the number is negative (sign bit)
    ///     exp: The biased exponent value (integer)
    ///     mantissa: The mantissa value (integer)
    #[new]
    fn new(
        sem: &Bound<'_, PyAny>,
        is_negative: bool,
        exp: i64,
        mantissa: u64,
    ) -> Self {
        let sem: PyRef<PySemantics> = sem.extract().unwrap();
        let mut man = BigInt::from_u64(mantissa);
        man.flip_bit(sem.inner.get_mantissa_len()); // Add the implicit bit.
        let bias = sem.inner.get_bias();
        PyFloat {
            inner: Float::from_parts(sem.inner, is_negative, exp - bias, man),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
    /// Returns the mantissa of the float.
    fn get_mantissa(&self) -> u64 {
        self.inner.get_mantissa().as_u64()
    }
    /// Returns the exponent of the float.
    fn get_exponent(&self) -> i64 {
        self.inner.get_exp()
    }
    /// Returns the category of the float.
    fn get_category(&self) -> String {
        format!("{:?}", self.inner.get_category())
    }
    /// Returns the semantics of the float.
    fn get_semantics(&self) -> PySemantics {
        PySemantics {
            inner: self.inner.get_semantics(),
        }
    }
    /// Get rounding mode of the number.
    fn get_rounding_mode(&self) -> String {
        self.inner.get_rounding_mode().as_string().to_string()
    }
    /// Returns true if the Float is negative
    fn is_negative(&self) -> bool {
        self.inner.is_negative()
    }
    /// Returns true if the Float is +-inf.
    fn is_inf(&self) -> bool {
        self.inner.is_inf()
    }
    /// Returns true if the Float is a +- NaN.
    fn is_nan(&self) -> bool {
        self.inner.is_nan()
    }
    /// Returns true if the Float is a +- zero.
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns true if this number is normal (not Zero, Nan, Inf).
    fn is_normal(&self) -> bool {
        self.inner.is_normal()
    }

    fn __add__(&self, other: &PyFloat) -> PyFloat {
        self.add(other)
    }

    fn __sub__(&self, other: &PyFloat) -> PyFloat {
        self.sub(other)
    }

    fn __mul__(&self, other: &PyFloat) -> PyFloat {
        self.mul(other)
    }
    fn __truediv__(&self, other: &PyFloat) -> PyFloat {
        self.div(other)
    }
    fn add(&self, other: &PyFloat) -> PyFloat {
        let val = self.inner.clone().add(other.inner.clone());
        PyFloat { inner: val }
    }
    fn mul(&self, other: &PyFloat) -> PyFloat {
        let val = self.inner.clone().mul(other.inner.clone());
        PyFloat { inner: val }
    }
    fn sub(&self, other: &PyFloat) -> PyFloat {
        let val = self.inner.clone().sub(other.inner.clone());
        PyFloat { inner: val }
    }
    fn div(&self, other: &PyFloat) -> PyFloat {
        let val = self.inner.clone().div(other.inner.clone());
        PyFloat { inner: val }
    }
    /// Returns the number raised to the power of `exp` which is an integer.
    fn powi(&self, exp: u64) -> PyFloat {
        PyFloat {
            inner: self.inner.powi(exp),
        }
    }
    /// Returns the number raised to the power of `exp` which is a float.
    fn pow(&self, exp: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.pow(&exp.inner),
        }
    }
    /// Returns the exponential of the number.
    fn exp(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.exp(),
        }
    }
    /// Returns the natural logarithm of the number.
    fn log(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.log(),
        }
    }
    /// Returns the sigmoid of the number.
    fn sigmoid(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.sigmoid(),
        }
    }
    /// Returns the absolute value of the number.
    fn abs(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.abs(),
        }
    }
    /// Returns the maximum of two numbers (as defined by IEEE 754).
    fn max(&self, other: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.max(&other.inner),
        }
    }
    /// Returns the minimum of two numbers (as defined by IEEE 754).
    fn min(&self, other: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.min(&other.inner),
        }
    }
    /// Returns the remainder of the division of two numbers.
    fn rem(&self, other: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.rem(&other.inner),
        }
    }
    /// Cast the number to another semantics.
    fn cast(&self, sem: &Bound<'_, PyAny>) -> PyFloat {
        let sem: PyRef<PySemantics> = sem.extract().unwrap();
        PyFloat {
            inner: self.inner.cast(sem.inner),
        }
    }
    /// Cast the number to another semantics with a specific rounding mode.
    fn cast_with_rm(&self, sem: &Bound<'_, PyAny>, rm: &str) -> PyFloat {
        let sem: PyRef<PySemantics> = sem.extract().unwrap();
        let rm = RoundingMode::from_string(rm);
        assert!(rm.is_some(), "Invalid rounding mode");
        PyFloat {
            inner: self.inner.cast_with_rm(sem.inner, rm.unwrap()),
        }
    }

    /// Returns the number with the sign flipped.
    fn neg(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.neg(),
        }
    }
    /// Returns the number with the sign flipped.
    fn __neg__(&self) -> PyFloat {
        self.neg()
    }
    /// Returns true if the number is less than the other number.
    fn __lt__(&self, other: &PyFloat) -> bool {
        self.inner < other.inner
    }
    /// Returns true if the number is less than or equal to the other number.
    fn __le__(&self, other: &PyFloat) -> bool {
        self.inner <= other.inner
    }
    /// Returns true if the number is equal to the other number.
    fn __eq__(&self, other: &PyFloat) -> bool {
        self.inner == other.inner
    }
    /// Returns true if the number is not equal to the other number.
    fn __ne__(&self, other: &PyFloat) -> bool {
        self.inner != other.inner
    }
    /// Returns true if the number is greater than the other number.
    fn __gt__(&self, other: &PyFloat) -> bool {
        self.inner > other.inner
    }
    /// Returns true if the number is greater than or equal to the other number.
    fn __ge__(&self, other: &PyFloat) -> bool {
        self.inner >= other.inner
    }
    /// Returns the sine of the number.
    fn sin(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.sin(),
        }
    }
    /// Returns the cosine of the number.
    fn cos(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.cos(),
        }
    }
    /// Returns the tangent of the number.
    fn tan(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.tan(),
        }
    }
    /// convert to f64.
    fn to_float64(&self) -> f64 {
        self.inner.as_f64()
    }
    /// Convert the number to a Continued Fraction of two integers.
    /// Take 'n' iterations.
    fn as_fraction(&self, n: usize) -> (u64, u64) {
        let (a, b) = self.inner.as_fraction(n);
        (a.as_u64(), b.as_u64())
    }
    /// Prints the number using the internal representation.
    fn dump(&self) {
        self.inner.dump();
    }
} // impl PyFloat

/// Returns the mathematical constant pi with the given semantics.
///
/// Args:
///     sem: The semantics to use for representing pi
#[pyfunction]
fn pi(sem: &Bound<'_, PyAny>) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::pi(sem.inner),
    })
}

/// Returns the fused multiply-add operation of three numbers.
///
/// Args: (a * b) + c
#[pyfunction]
fn fma(a: &PyFloat, b: &PyFloat, c: &PyFloat) -> PyResult<PyFloat> {
    Ok(PyFloat {
        inner: Float::fma(&a.inner, &b.inner, &c.inner),
    })
}

/// Returns the mathematical constant e (Euler's number) with the given semantics.
///
/// Args:
///     sem: The semantics to use for representing e
#[pyfunction]
fn e(sem: &Bound<'_, PyAny>) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::e(sem.inner),
    })
}

/// Returns the natural logarithm of 2 (ln(2)) with the given semantics.
///
/// Args:
///     sem: The semantics to use for representing ln(2)
#[pyfunction]
fn ln2(sem: &Bound<'_, PyAny>) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::ln2(sem.inner),
    })
}

/// Returns the number zero with the given semantics.
///
/// Args:
///     sem: The semantics to use for representing e
#[pyfunction]
fn zero(sem: &Bound<'_, PyAny>) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::zero(sem.inner, false),
    })
}

/// Returns a new float with the integer value 'val' with the given semantics.
///
/// Args:
///     sem: The semantics to use
///     val: The integer value
#[pyfunction]
fn from_i64(sem: &Bound<'_, PyAny>, val: i64) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::from_i64(sem.inner, val),
    })
}

/// Returns a new float with the fp64 value 'val'.
///
/// Args:
///     val: The f64 value
#[pyfunction]
fn from_fp64(val: f64) -> PyResult<PyFloat> {
    Ok(PyFloat {
        inner: Float::from_f64(val),
    })
}

#[pymodule]
fn _arpfloat(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFloat>()?;
    m.add_class::<PySemantics>()?;

    // Add the functions to the module
    m.add_function(wrap_pyfunction!(pi, m)?)?;
    m.add_function(wrap_pyfunction!(e, m)?)?;
    m.add_function(wrap_pyfunction!(ln2, m)?)?;
    m.add_function(wrap_pyfunction!(zero, m)?)?;
    m.add_function(wrap_pyfunction!(fma, m)?)?;
    m.add_function(wrap_pyfunction!(from_i64, m)?)?;
    m.add_function(wrap_pyfunction!(from_fp64, m)?)?;
    Ok(())
}
