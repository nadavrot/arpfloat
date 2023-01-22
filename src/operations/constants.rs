//! This module contains the implementation of methods that compute mathematical
//! constants.
//!
use crate::RoundingMode;
use crate::{Float, Semantics};

impl Float {
    /// Computes pi.
    pub fn pi(sem: Semantics) -> Self {
        // Algorithm description in Pg 246:
        // Fast Multiple-Precision Evaluation of Elementary Functions
        // by Richard P. Brent.

        // Increase the precision, because the arithmetic operations below
        // require rounding, so if we want to get the accurate results we need
        // to operate with increased precision.
        let orig_sem = sem;
        let sem = sem.grow_log(4);

        use RoundingMode::NearestTiesToEven as rm;

        let one = Self::from_i64(sem, 1);
        let two = Self::from_i64(sem, 2);
        let four = Self::from_i64(sem, 4);

        let mut a = one.clone();
        let mut b = one.clone() / two.sqrt();
        let mut t = one.clone() / four;
        let mut x = one;

        while a != b {
            let y = a.clone();
            a = (&a + &b).scale(-1, rm);
            b = (&b * &y).sqrt();
            t -= &x * (&a - &y).sqr();
            x = x.scale(1, rm);
        }
        (a.sqr() / t).cast(orig_sem)
    }

    /// Computes e.
    pub fn e(sem: Semantics) -> Self {
        let orig_sem = sem;
        let sem = sem.increase_precision(1);

        let one = Self::one(sem, false);
        let mut term = one.clone();

        // Use Euler's continued fraction, which is a simple series.
        let iterations: i64 = (sem.get_exponent_len() * 2) as i64;
        for i in (1..iterations).rev() {
            let v = Self::from_i64(sem, i);
            term = &v + &v / &term;
        }

        (one / term + 2).cast(orig_sem)
    }

    /// Compute log(2).
    pub fn ln2(sem: Semantics) -> Self {
        use RoundingMode::None as rm;
        let sem2 = sem.increase_precision(8);

        // Represent log(2) using the sum 1/k*2^k
        let one = Self::one(sem2, false);
        let mut sum = Self::zero(sem2, false);
        let mut prev = Self::inf(sem2, true);
        for k in 1..500 {
            let k2 = Self::from_u64(sem2, 1).scale(k, rm);
            let k = Self::from_u64(sem2, k as u64);
            let kk2 = &Float::mul_with_rm(&k, &k2, rm);
            let term = Float::div_with_rm(&one, kk2, rm);
            sum = Float::add_with_rm(&sum, &term, rm);
            if prev == sum {
                break;
            }
            prev = sum.clone();
        }
        sum.cast(sem)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_pi() {
    use crate::FP32;
    use crate::FP64;
    assert_eq!(Float::pi(FP64).as_f64(), std::f64::consts::PI);
    assert_eq!(Float::pi(FP32).as_f32(), std::f32::consts::PI);
}

#[cfg(feature = "std")]
#[test]
fn test_e() {
    use crate::FP32;
    use crate::FP64;
    assert_eq!(Float::e(FP64).as_f64(), std::f64::consts::E);
    assert_eq!(Float::e(FP32).as_f32(), std::f32::consts::E);
}

#[cfg(feature = "std")]
#[test]
fn test_ln2() {
    use crate::FP64;
    assert_eq!(Float::ln2(FP64).as_f64(), std::f64::consts::LN_2);
}
