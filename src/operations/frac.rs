//! This module contains the implementation of methods that compute continued
//! fraction.

use crate::{bigint::BigInt, Float};

impl Float {
    /// Convert the number to a Continued Fraction of two integers.
    /// The fraction is computed using 'n' iterations of the form:
    /// a0 + 1/(a1 + 1/(a2 + 1/( ... ))).
    /// This method discards the sign, and returns (0, 0) for Inf and NaN.
    pub fn as_fraction(&self, n: usize) -> (BigInt, BigInt) {
        if self.is_zero() {
            return (BigInt::zero(), BigInt::one()); // Zero.
        } else if self.is_inf() || self.is_nan() {
            return (BigInt::zero(), BigInt::zero()); // Invalid.
        }

        // Algorithm from:
        // Elementary Functions: Algorithms and Implementation
        // 9.3.1 A few basic notions on continued fractions - Page 180.
        extern crate alloc;
        use alloc::vec::Vec;
        let sem = self.get_semantics();
        let rm = sem.get_rounding_mode();

        let one = Self::one(sem, false);
        let mut real = self.clone();
        let mut a: Vec<BigInt> = Vec::new();

        for _ in 0..n.max(2) {
            let int = real.trunc();
            a.push(int.convert_normal_to_integer(rm));
            real = &one / (real - int);
        }

        let one = BigInt::one();
        let mut p = (&one + &(&a[0] * &a[1]), a[0].clone());
        let mut q = (a[1].clone(), one);

        if n < 2 {
            return (p.1, q.1);
        }

        for elem in a.iter().skip(2) {
            p = (&p.1 + &(elem * &p.0), p.0);
            q = (&q.1 + &(elem * &q.0), q.0);
        }

        (p.0, q.0)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_frac() {
    use crate::FP128;
    let x = Float::pi(FP128);

    // Verified with https://oeis.org/A001203.
    let (p, q) = x.as_fraction(1);
    assert_eq!((3, 1), (p.as_u64(), q.as_u64()));
    let (p, q) = x.as_fraction(2);
    assert_eq!((22, 7), (p.as_u64(), q.as_u64()));
    let (p, q) = x.as_fraction(3);
    assert_eq!((333, 106), (p.as_u64(), q.as_u64()));
    let (p, q) = x.as_fraction(4);
    assert_eq!((355, 113), (p.as_u64(), q.as_u64()));
}
