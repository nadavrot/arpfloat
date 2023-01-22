//! This module contains the implementation of log- and exp-related methods.
//!
use crate::RoundingMode;

use crate::float::Float;

impl Float {
    /// Computes the taylor series, centered around 1, and valid in [0..2].
    /// z = (x - 1)/(x + 1)
    /// log(x) = 2 (z + z^3/3 + z^5/5 + z^7/7 ... )
    fn log_taylor(x: &Self) -> Self {
        use RoundingMode::None as rm;
        let sem = x.get_semantics();
        let one = Self::one(sem, false);
        let up = Float::sub_with_rm(x, &one, rm);
        let down = Float::add_with_rm(x, &one, rm);
        let z = Float::div_with_rm(&up, &down, rm);
        let z2 = z.sqr();

        let mut top = z;
        let mut sum = Self::zero(sem, false);
        let mut prev = Self::one(sem, true);
        for i in 0..50 {
            if prev == sum {
                break; // Stop if we are not making progress.
            }
            prev = sum.clone();

            let bottom = &Self::from_u64(sem, i * 2 + 1);
            let elem = Float::div_with_rm(&top, bottom, rm);
            sum = Float::add_with_rm(&sum, &elem, rm);

            // Prepare the next iteration.
            top = Float::mul_with_rm(&top, &z2, rm);
        }

        sum.scale(1, RoundingMode::Zero)
    }

    /// Reduce the range of 'x' with the identity:
    /// ln(x) = ln(sqrt(x)^2) = 2 * ln(sqrt(x)) and
    /// ln(x) = -ln(1/x)
    fn log_range_reduce(x: &Self) -> Self {
        use RoundingMode::NearestTiesToEven as even;
        let sem = x.get_semantics();
        let up = Self::from_f64(1.001).cast(sem);
        let one = Self::from_u64(sem, 1);

        if x > &up {
            let sx = x.sqrt();
            return Self::log_range_reduce(&sx).scale(1, even);
        }

        if x < &one {
            let re = Float::div_with_rm(&one, x, RoundingMode::None);
            return Self::log_range_reduce(&re).neg();
        }

        Self::log_taylor(x)
    }

    /// Computes logarithm of 'x'.
    pub fn log(&self) -> Self {
        use RoundingMode::None as rm;
        let sem = self.get_semantics();

        //Fast Logarithm function for Arbitrary Precision number,
        // by Henrik Vestermark.

        // Handle all of the special cases:
        if !self.is_normal() || self.is_negative() {
            return Self::nan(sem, self.get_sign());
        }

        let orig_sem = self.get_semantics();
        let sem = orig_sem.grow_log(10).increase_exponent(10);

        let x = &self.cast_with_rm(sem, rm);
        Self::log_range_reduce(x).cast_with_rm(orig_sem, rm)
    }
}

#[test]
fn test_log() {
    use crate::FP128;
    let x = Float::from_f64(0.1).cast(FP128).log();
    assert_eq!(x.as_f64(), -2.3025850929940455);

    for x in [
        0.1, 0.5, 2.3, 4.5, 9.8, 11.2, 15.2, 91.2, 102.2, 192.4, 1024.2,
        90210.2,
    ] {
        let lhs = Float::from_f64(x).cast(FP128).log().as_f64();
        let rhs = x.ln();
        assert_eq!(lhs, rhs);
    }
}

impl Float {
    /// Computes the taylor series:
    /// exp(x) = 1 + x/1! + x^2/2! + x^3/3! ...
    fn exp_taylor(x: &Self) -> Self {
        let sem = x.get_semantics();
        use crate::bigint::BigInt;
        let mut top = Self::one(sem, false);
        let mut bottom = BigInt::one();

        let mut sum = Self::zero(sem, false);
        let mut prev = Self::one(sem, true);
        for k in 1..50 {
            if prev == sum {
                break; // Stop if we are not making progress.
            }
            prev = sum.clone();

            let elem = &top / &Self::from_bigint(sem, bottom.clone());
            sum += elem;

            // Prepare the next iteration.
            bottom *= BigInt::from_u64(k);
            top = &top * x;
        }

        sum
    }

    /// Reduce the range of 'x' with the identity:
    /// e^x = (e^(x/2))^2
    fn exp_range_reduce(x: &Self) -> Self {
        let sem = x.get_semantics();

        let one = Self::from_u64(sem, 1);

        if x > &one {
            let sx = x.scale(-3, RoundingMode::Zero);
            let esx = Self::exp_range_reduce(&sx);
            return esx.sqr().sqr().sqr();
        }

        Self::exp_taylor(x)
    }

    /// Computes exponential function `e^self`.
    pub fn exp(&self) -> Self {
        let sem = self.get_semantics();

        // Handle all of the special cases:
        if self.is_zero() {
            return Self::one(sem, false);
        } else if !self.is_normal() {
            return Self::nan(sem, self.get_sign());
        }

        let orig_sem = self.get_semantics();
        let sem = orig_sem.grow_log(10).increase_exponent(10);

        // Handle the negative values.
        if self.is_negative() {
            let one = Self::one(sem, false);
            return (one / self.cast(sem).neg().exp()).cast(orig_sem);
        }

        Self::exp_range_reduce(&self.cast(sem)).cast(orig_sem)
    }
}

#[test]
fn test_exp() {
    assert_eq!(Float::from_f64(2.51).exp().as_f64(), 12.30493006051041);

    for x in [
        0.000003, 0.001, 0.12, 0.13, 0.5, 1.2, 2.3, 4.5, 9.8, 5.0, 11.2, 15.2,
        25.0, 34.001, 54., 89.1, 91.2, 102.2, 150., 192.4, 212., 256., 102.3,
    ] {
        let lhs = Float::from_f64(x).exp().as_f64();
        let rhs = x.exp();
        assert_eq!(lhs, rhs);
    }
}
