//! This module contains the implementation of several arithmetic operations.

use crate::RoundingMode;

use crate::float::Float;

impl Float {
    /// Return this number raised to the power of 'n'.
    pub fn powi(&self, mut n: u64) -> Self {
        let sem = self.get_semantics().increase_precision(2);
        let mut elem = Self::one(sem, false);
        // This algorithm is similar to binary conversion. Each bit in 'n'
        // represents a power-of-two number, like 1,2,4,8 ... We know how to
        // generate numbers to the power of an even number by squaring the
        // number log2 times. So, we just multiply all of the numbers together
        // to get the result. This is like converting a binary number to integer
        // except that instead of adding we multiply the values.
        let mut val = self.cast(sem);
        while n > 0 {
            if n & 1 == 1 {
                elem *= &val;
            }
            val *= &val.clone();
            n >>= 1;
        }
        elem.cast(self.get_semantics())
    }

    /// Calculates the power of two.
    pub fn sqr(&self) -> Self {
        self.powi(2)
    }
    /// Calculates the square root of the number.
    pub fn sqrt(&self) -> Self {
        let sem = self.get_semantics();
        if self.is_zero() {
            return self.clone(); // (+/-) zero
        } else if self.is_nan() || self.is_negative() {
            return Self::nan(sem, self.get_sign()); // (-/+)Nan, -Number.
        } else if self.is_inf() {
            return self.clone(); // Inf+.
        }

        let target = self.clone();
        let two = Self::from_u64(sem, 2);

        // Start the search at max(2, x).
        let mut x = if target < two { two } else { target.clone() };
        let mut prev = x.clone();

        // Use the Newton Raphson method.
        loop {
            x += &target / &x;
            x = x.scale(-1, RoundingMode::NearestTiesToEven);
            // Stop when value did not change or regressed.
            if prev < x || x == prev {
                return x;
            }
            prev = x.clone();
        }
    }

    /// Returns the absolute value of this float.
    pub fn abs(&self) -> Self {
        let mut x = self.clone();
        x.set_sign(false);
        x
    }

    /// Returns the greater of self and `other`.
    pub fn max(&self, other: &Self) -> Self {
        if self.is_nan() {
            return other.clone();
        } else if other.is_nan() {
            return self.clone();
        } else if self.get_sign() != other.get_sign() {
            return if self.get_sign() {
                other.clone()
            } else {
                self.clone()
            }; // Handle (+-)0.
        }
        if self > other {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Returns the smaller of self and `other`.
    pub fn min(&self, other: &Self) -> Self {
        if self.is_nan() {
            return other.clone();
        } else if other.is_nan() {
            return self.clone();
        } else if self.get_sign() != other.get_sign() {
            return if self.get_sign() {
                self.clone()
            } else {
                other.clone()
            }; // Handle (+-)0.
        }
        if self > other {
            other.clone()
        } else {
            self.clone()
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn test_sqrt() {
    use crate::utils;
    use crate::FP64;

    // Try a few power-of-two values.
    for i in 0..256 {
        let v16 = Float::from_u64(FP64, i * i);
        assert_eq!(v16.sqrt().as_f64(), (i) as f64);
    }

    // Test the category and value of the different special values (inf, zero,
    // correct sign, etc).
    for v_f64 in utils::get_special_test_values() {
        let vf = Float::from_f64(v_f64);
        assert_eq!(vf.sqrt().is_inf(), v_f64.sqrt().is_infinite());
        assert_eq!(vf.sqrt().is_nan(), v_f64.sqrt().is_nan());
        assert_eq!(vf.sqrt().is_negative(), v_f64.sqrt().is_sign_negative());
    }

    // Test precomputed values.
    fn check(inp: f64, res: f64) {
        assert_eq!(Float::from_f64(inp).sqrt().as_f64(), res);
    }
    check(1.5, 1.224744871391589);
    check(2.3, 1.51657508881031);
    check(6.7, 2.588435821108957);
    check(7.9, 2.8106938645110393);
    check(11.45, 3.383784863137726);
    check(1049.3, 32.39290045673589);
    check(90210.7, 300.35096137685326);
    check(199120056003.73413, 446228.70369770494);
    check(0.6666666666666666, 0.816496580927726);
    check(0.4347826086956522, 0.6593804733957871);
    check(0.14925373134328357, 0.3863337046431279);
    check(0.12658227848101264, 0.35578403348241);
    check(0.08733624454148473, 0.29552706228277087);
    check(0.0009530162965786716, 0.030870962028719993);
    check(1.1085159520988087e-5, 0.00332943831914455);
    check(5.0120298432056786e-8, 0.0002238756316173263);
}

#[cfg(feature = "std")]
#[test]
fn test_min_max() {
    use crate::utils;

    fn check(v0: f64, v1: f64) {
        // Min.
        let correct = v0.min(v1);
        let test = Float::from_f64(v0).min(&Float::from_f64(v1)).as_f64();
        assert_eq!(test.is_nan(), correct.is_nan());
        if !correct.is_nan() {
            assert_eq!(correct, test);
        }
        // Max.
        let correct = v0.max(v1);
        let test = Float::from_f64(v0).max(&Float::from_f64(v1)).as_f64();
        assert_eq!(test.is_nan(), correct.is_nan());
        if !correct.is_nan() {
            assert_eq!(correct, test);
        }
    }

    // Test a bunch of special values (Inf, Epsilon, Nan, (+-)Zeros).
    for v0 in utils::get_special_test_values() {
        for v1 in utils::get_special_test_values() {
            check(v0, v1);
        }
    }

    let mut lfsr = utils::Lfsr::new();

    for _ in 0..100 {
        let v0 = f64::from_bits(lfsr.get64());
        let v1 = f64::from_bits(lfsr.get64());
        check(v0, v1);
    }
}

#[cfg(feature = "std")]
#[test]
fn test_abs() {
    use crate::utils;

    for v in utils::get_special_test_values() {
        if !v.is_nan() {
            assert_eq!(Float::from_f64(v).abs().as_f64(), v.abs());
        }
    }
}

//  Compute basic constants.

impl Float {
    /// Similar to 'scalbln'. Adds or subtracts to the exponent of the number,
    /// and scaling it by 2^exp.
    pub fn scale(&self, scale: i64, rm: RoundingMode) -> Self {
        use crate::bigint::LossFraction;
        if !self.is_normal() {
            return self.clone();
        }

        let mut r = Self::from_parts(
            self.get_semantics(),
            self.get_sign(),
            self.get_exp() + scale,
            self.get_mantissa(),
        );
        r.normalize(rm, LossFraction::ExactlyZero);
        r
    }

    /// Returns the remainder from a division of two floats. This is equivalent
    /// to rust 'rem' or c 'fmod'.
    pub fn rem(&self, rhs: &Self) -> Self {
        use core::ops::Sub;
        // Handle NaNs.
        if self.is_nan() || rhs.is_nan() || self.is_inf() || rhs.is_zero() {
            return Self::nan(self.get_semantics(), self.get_sign());
        }
        // Handle values that are obviously zero or self.
        if self.is_zero() || rhs.is_inf() {
            return self.clone();
        }

        // Operate on integers.
        let mut lhs = self.abs();
        let rhs = if rhs.is_negative() {
            rhs.neg()
        } else {
            rhs.clone()
        };
        debug_assert!(lhs.is_normal() && rhs.is_normal());

        // This is a clever algorithm. Subtracting the RHS from LHS in a loop
        // would be slow, but we perform a divide-like algorithm where we shift
        // 'rhs' by higher powers of two, and subtract it from LHS, until LHS is
        // lower than RHS.
        while lhs >= rhs && lhs.is_normal() {
            let scale = lhs.get_exp() - rhs.get_exp();

            // Scale RHS by a power of two. If we overshoot, take a step back.
            let mut diff = rhs.scale(scale, RoundingMode::None);
            if diff > lhs {
                diff = rhs.scale(scale - 1, RoundingMode::None);
            }

            lhs = lhs.sub(diff);
        }

        // Set the original sign.
        lhs.set_sign(self.get_sign());
        lhs
    }
}

#[test]
fn test_scale() {
    use crate::FP64;
    let x = Float::from_u64(FP64, 1);
    let y = x.scale(1, RoundingMode::None);
    assert_eq!(y.as_f64(), 2.0);
    let z = x.scale(-1, RoundingMode::None);
    assert_eq!(z.as_f64(), 0.5);
}

#[cfg(feature = "std")]
#[test]
fn test_rem() {
    use crate::utils;
    use crate::utils::Lfsr;

    use core::ops::Rem;

    fn check_two_numbers(v0: f64, v1: f64) {
        let f0 = Float::from_f64(v0);
        let f1 = Float::from_f64(v1);
        let r0 = v0.rem(v1);
        let r1 = f0.rem(&f1).as_f64();
        assert_eq!(r0.is_nan(), r1.is_nan());
        if !r0.is_nan() {
            assert_eq!(r0, r1);
        }
    }

    // Test addition, multiplication, subtraction with random values.
    check_two_numbers(1.4, 2.5);
    check_two_numbers(2.4, 1.5);
    check_two_numbers(1000., std::f64::consts::PI);
    check_two_numbers(10000000000000000000., std::f64::consts::PI / 1000.);
    check_two_numbers(10000000000000000000., std::f64::consts::PI);
    check_two_numbers(100., std::f64::consts::PI);
    check_two_numbers(100., -std::f64::consts::PI);
    check_two_numbers(0., 10.);
    check_two_numbers(std::f64::consts::PI, 10.0);

    // Test a bunch of random values:
    let mut lfsr = Lfsr::new();
    for _ in 0..5000 {
        let v0 = f64::from_bits(lfsr.get64());
        let v1 = f64::from_bits(lfsr.get64());
        check_two_numbers(v0, v1);
    }

    // Test the hard cases:
    for v0 in utils::get_special_test_values() {
        for v1 in utils::get_special_test_values() {
            check_two_numbers(v0, v1);
        }
    }
}

#[test]
fn test_powi() {
    assert_eq!(Float::from_f64(2.).powi(0).as_f64(), 1.);
    assert_eq!(Float::from_f64(2.).powi(1).as_f64(), 2.);
    assert_eq!(Float::from_f64(2.).powi(3).as_f64(), 8.);
    assert_eq!(Float::from_f64(2.).powi(5).as_f64(), 32.);
    assert_eq!(Float::from_f64(2.).powi(10).as_f64(), 1024.);
    assert_eq!(Float::from_f64(0.3).powi(3).as_f64(), 0.026999999999999996);
}

impl Float {
    /// Return this number raised to the power of 'n'.
    /// Computed using e^(n * log(self))
    pub fn pow(&self, n: &Float) -> Self {
        let orig_sem = self.get_semantics();
        let one = Self::one(orig_sem, false);
        let sign = self.get_sign();

        assert_eq!(orig_sem, n.get_semantics());

        if *self == one {
            return self.clone();
        } else if n.is_inf() || n.is_nan() {
            return Self::nan(orig_sem, sign);
        } else if n.is_zero() {
            return Self::one(orig_sem, sign);
        } else if self.is_zero() {
            return if n.is_negative() {
                Self::inf(orig_sem, sign)
            } else {
                Self::zero(orig_sem, sign)
            };
        } else if self.is_negative() || self.is_inf() || self.is_nan() {
            return Self::nan(orig_sem, sign);
        }

        let sem = orig_sem.grow_log(10).increase_exponent(10);
        (n.cast(sem) * self.cast(sem).log()).exp().cast(orig_sem)
    }
}

#[test]
fn test_pow() {
    fn my_pow(a: f32, b: f32) -> f32 {
        Float::from_f32(a).pow(&Float::from_f32(b)).as_f32()
    }

    assert_eq!(my_pow(1.24, 1.2), 1.2945118);
    assert_eq!(my_pow(0.94, 13.), 0.44736509);
    assert_eq!(my_pow(0.11, -8.), 46650738.02097334);
    assert_eq!(my_pow(40.0, 3.1), 92552.0);

    for i in 0..30 {
        for j in -10..10 {
            let i = i as f64;
            let j = j as f64;
            let res = i.powf(j);
            let res2 = Float::from_f64(i).pow(&Float::from_f64(j));
            assert_eq!(res, res2.as_f64());
        }
    }
}
