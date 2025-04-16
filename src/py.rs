use core::ops::Add;
use pyo3::prelude::*;
use std::format;
use std::string::String;
use std::string::ToString;

use crate::{BigInt, Float, RoundingMode, Semantics};

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
        // parse the rounding mode string
        let rm = match rounding_mode_str {
            "NearestTiesToEven" => RoundingMode::NearestTiesToEven,
            "NearestTiesToAway" => RoundingMode::NearestTiesToAway,
            "Zero" => RoundingMode::Zero,
            "Positive" => RoundingMode::Positive,
            "Negative" => RoundingMode::Negative,
            _ => panic!("Invalid rounding mode string"),
        };

        let sem = Semantics::new(exp_size as usize, mantissa_size as usize, rm);
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
    fn __str__(&self) -> String {
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

    fn add(&self, other: &PyFloat) -> PyFloat {
        let val = self.inner.clone().add(other.inner.clone());
        PyFloat { inner: val }
    }

    fn sqrt(&self) -> PyFloat {
        PyFloat {
            inner: self.inner.sqrt(),
        }
    }

    fn to_float64(&self) -> f64 {
        self.inner.as_f64()
    }

    fn dump(&self) {
        self.inner.dump();
    }
} // impl PyFloat

#[pymodule]
fn _arpfloat(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFloat>()?;
    m.add_class::<PySemantics>()?;
    Ok(())
}
