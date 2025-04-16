use crate::{BigInt, Float, RoundingMode, Semantics};
use core::ops::{Add, Div, Mul, Sub};
use pyo3::prelude::*;
use std::format;
use std::string::String;
use std::string::ToString;

#[pyclass]
struct PySemantics {
    inner: Semantics,
}

#[pymethods]
impl PySemantics {
    #[new]
    fn py_new(
        exp_size: i64,
        mantissa_size: u64,
        rounding_mode_str: &str,
    ) -> Self {
        let rm = RoundingMode::from_string(rounding_mode_str);
        assert!(rm.is_some(), "Invalid rounding mode");
        let sem = Semantics::new(
            exp_size as usize,
            mantissa_size as usize,
            rm.unwrap(),
        );
        PySemantics { inner: sem }
    }
    fn get_exponent_len(&self) -> usize {
        self.inner.get_exponent_len()
    }
    fn get_mantissa_len(&self) -> usize {
        self.inner.get_mantissa_len()
    }
    fn get_rounding_mode(&self) -> String {
        self.inner.get_rounding_mode().as_string().to_string()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
struct PyFloat {
    inner: Float,
}

#[pymethods]
impl PyFloat {
    #[new]
    fn py_new(
        sem: &Bound<'_, PyAny>,
        is_negative: bool,
        exp: i64,
        mantissa: u64,
    ) -> Self {
        let sem: PyRef<PySemantics> = sem.extract().unwrap();
        let mantissa = BigInt::from_u64(mantissa);
        PyFloat {
            inner: Float::new(sem.inner, is_negative, exp, mantissa),
        }
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
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
    fn powi(&self, exp: u64) -> PyFloat {
        PyFloat {
            inner: self.inner.powi(exp),
        }
    }
    fn pow(&self, exp: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.pow(&exp.inner),
        }
    }
    fn exp(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.exp(),
        }
    }
    fn log(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.log(),
        }
    }
    fn sigmoid(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.sigmoid(),
        }
    }
    fn abs(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.abs(),
        }
    }
    fn max(&self, other: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.max(&other.inner),
        }
    }
    fn min(&self, other: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.min(&other.inner),
        }
    }
    fn rem(&self, other: &PyFloat) -> PyFloat {
        PyFloat {
            inner: self.inner.rem(&other.inner),
        }
    }
    fn cast(&self, sem: &Bound<'_, PyAny>) -> PyFloat {
        let sem: PyRef<PySemantics> = sem.extract().unwrap();
        PyFloat {
            inner: self.inner.cast(sem.inner),
        }
    }
    fn cast_with_rm(&self, sem: &Bound<'_, PyAny>, rm: &str) -> PyFloat {
        let sem: PyRef<PySemantics> = sem.extract().unwrap();
        let rm = RoundingMode::from_string(rm);
        assert!(rm.is_some(), "Invalid rounding mode");
        PyFloat {
            inner: self.inner.cast_with_rm(sem.inner, rm.unwrap()),
        }
    }
    fn sin(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.sin(),
        }
    }
    fn cos(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.cos(),
        }
    }
    fn tan(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.tan(),
        }
    }
    fn to_float64(&self) -> f64 {
        self.inner.as_f64()
    }
    fn dump(&self) {
        self.inner.dump();
    }
} // impl PyFloat

#[pyfunction]
fn pi(sem: &Bound<'_, PyAny>) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::pi(sem.inner),
    })
}

#[pyfunction]
fn e(sem: &Bound<'_, PyAny>) -> PyResult<PyFloat> {
    let sem: PyRef<PySemantics> = sem.extract()?;
    Ok(PyFloat {
        inner: Float::e(sem.inner),
    })
}

#[pymodule]
fn _arpfloat(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFloat>()?;
    m.add_class::<PySemantics>()?;

    // Add the functions to the module
    m.add_function(wrap_pyfunction!(pi, m)?)?;
    m.add_function(wrap_pyfunction!(e, m)?)?;

    Ok(())
}
