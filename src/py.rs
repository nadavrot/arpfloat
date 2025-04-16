use core::ops::Add;

// Use pyo3 to wrap the library for python.
use pyo3::prelude::*;

use crate::{BigInt, Float, RoundingMode, Semantics};

#[pyclass]
struct PyFloat {
    inner: Float,
}

#[pymethods]
impl PyFloat {
    #[new]
    fn py_new(
        exp_size: i64,
        mantissa_size: u64,
        sign: bool,
        exp: i64,
        mantissa: u64,
    ) -> Self {
        let sem = Semantics::new(
            exp_size as usize,
            mantissa_size as usize,
            RoundingMode::NearestTiesToEven,
        );
        let mantissa = BigInt::from_u64(mantissa);
        PyFloat {
            inner: Float::new(sem, sign, exp, mantissa),
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
    Ok(())
}
