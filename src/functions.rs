use crate::{float::Semantics, RoundingMode};

use super::float::Float;

impl Float {
    /// Return this number raised to the power of 'n'.
    pub fn powi(&self, mut n: u64) -> Self {
        use RoundingMode::NearestTiesToEven as rm;
        let mut elem = Self::one(self.get_semantics(), false);
        // This algorithm is similar to binary conversion. Each bit in 'n'
        // represents a power-of-two number, like 1,2,4,8 ... We know how to
        // generate numbers to the power of an even number by squaring the
        // number log2 times. So, we just multiply all of the numbers together
        // to get the result. This is like converting a binary number to integer
        // except that instead of adding we multiply the values.
        let mut val = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                elem = Self::mul_with_rm(&elem, &val, rm);
            }
            val = Self::mul_with_rm(&val, &val, rm);
            n >>= 1;
        }
        elem
    }

    /// Calculates the power of two.
    pub fn sqr(&self) -> Self {
        self.powi(2)
    }
    /// Calculates the square root of the number using the Newton Raphson
    /// method.
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
    use super::utils;
    use super::FP64;

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
    use super::utils;

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
    use super::utils;

    for v in utils::get_special_test_values() {
        if !v.is_nan() {
            assert_eq!(Float::from_f64(v).abs().as_f64(), v.abs());
        }
    }
}

//  Compute basic constants.

impl Float {
    /// Computes PI -- Algorithm description in Pg 246:
    /// Fast Multiple-Precision Evaluation of Elementary Functions
    /// by Richard P. Brent.
    pub fn pi(sem: Semantics) -> Self {
        // Increase the precision, because the arithmetic operations below
        // require rounding, so if we want to get the accurate results we need
        // to operate with increased precision.
        let orig_sem = sem;
        let sem = sem.increase_precision(10).increase_exponent(4);

        let one = Self::from_i64(sem, 1);
        let two = Self::from_i64(sem, 2);
        let four = Self::from_i64(sem, 4);

        let mut a = one.clone();
        let mut b = one.clone() / two.sqrt();
        let mut t = one.clone() / four;
        let mut x = one;

        while a != b {
            let y = a.clone();
            a = (&a + &b) / 2;
            b = (&b * &y).sqrt();
            t = &t - (&x * (&a - &y).sqr());
            x = &x * 2;
        }
        (a.sqr() / t).cast(orig_sem)
    }

    /// Computes e using Euler's continued fraction, which is a simple series.
    pub fn e(sem: Semantics) -> Self {
        let orig_sem = sem;
        let sem = sem.increase_precision(10);

        let one = Self::from_i64(sem, 1);
        let mut term = one.clone();
        let iterations: i64 = (sem.get_exponent_len() * 2) as i64;
        for i in (1..iterations).rev() {
            let v = Self::from_i64(sem, i);
            term = &v + &v / &term;
        }

        (one / term + 2).cast(orig_sem)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_pi() {
    use crate::FP64;
    assert_eq!(Float::pi(FP64).as_f64(), std::f64::consts::PI);
}

#[cfg(feature = "std")]
#[test]
fn test_e() {
    use super::FP32;
    use super::FP64;
    assert_eq!(Float::e(FP64).as_f64(), std::f64::consts::E);
    assert_eq!(Float::e(FP32).as_f32(), std::f32::consts::E);
}

impl Float {
    /// Similar to 'scalbln'. Adds or subtracts to the exponent of the number,
    /// and scaling it by 2^exp.
    pub fn scale(&self, scale: i64, rm: RoundingMode) -> Self {
        use crate::bigint::LossFraction;
        if !self.is_normal() {
            return self.clone();
        }

        let mut r = Self::new(
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
            let mut diff = rhs.scale(scale, RoundingMode::NearestTiesToEven);
            if diff > lhs {
                diff = rhs.scale(scale - 1, RoundingMode::NearestTiesToEven);
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
    use super::FP64;
    let x = Float::from_u64(FP64, 1);
    let y = x.scale(1, RoundingMode::NearestTiesToEven);
    assert_eq!(y.as_f64(), 2.0);
    let z = x.scale(-1, RoundingMode::NearestTiesToEven);
    assert_eq!(z.as_f64(), 0.5);
}

#[cfg(feature = "std")]
#[test]
fn test_rem() {
    use super::utils;
    use super::utils::Lfsr;

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

impl Float {
    /// sin(x) = x - x^3 / 3! + x^5 / 5! - x^7/7! ....
    fn sin_taylor(x: &Self) -> Self {
        use crate::bigint::BigInt;
        let sem = x.get_semantics();

        let mut neg = false;
        let mut top = x.clone();
        let mut bottom = BigInt::one();
        let mut sum = Self::zero(sem, false);
        let x2 = x.sqr();
        let mut prev = Self::one(sem, true);
        for i in 1..50 {
            if prev == sum {
                break; // Stop if we are not making progress.
            }
            prev = sum.clone();
            // Update sum.
            let elem = &top / &Self::from_bigint(sem, bottom.clone());
            sum = if neg { sum - elem } else { sum + elem };

            // Prepare the next element.
            top = &top * &x2;
            let next_term = BigInt::from_u64((i * 2) * (i * 2 + 1));
            bottom = bottom * &next_term;
            neg ^= true;
        }

        sum
    }

    /// Reduce sin(x) in the range 0..pi/2, using the identity:
    /// sin(3x) = 3sin(x)-4(sin(x)^3)
    fn sin_step4_reduction(x: &Self, steps: usize) -> Self {
        if steps == 0 {
            return Self::sin_taylor(x);
        }

        let x3 = x / 3;
        let sx = Self::sin_step4_reduction(&x3, steps - 1);
        (&sx * 3) - sx.powi(3) * 4
    }

    /// Return the sine function.
    pub fn sin(&self) -> Self {
        // Fast Trigonometric functions for Arbitrary Precision number
        // by Henrik Vestermark.

        if self.is_zero() || self.is_nan() {
            return self.clone();
        }

        if self.is_inf() {
            return Self::nan(self.get_semantics(), self.get_sign());
        }

        let orig_sem = self.get_semantics();
        let sem = orig_sem.increase_precision(20).increase_exponent(4);

        assert!(self.is_normal());

        let mut neg = false;
        // Step1 range reduction.

        let mut val = self.cast(sem);

        // Handle the negatives.
        if val.is_negative() {
            val = val.neg();
            neg ^= true;
        }
        let pi = Self::pi(sem);
        let pi2 = pi.scale(1, RoundingMode::Zero);
        let pi_half = pi.scale(-1, RoundingMode::Zero);

        // Step 1
        if val > pi2 {
            val = val.rem(&pi2);
        }

        debug_assert!(val <= pi2);
        // Step 2.
        if val > pi {
            val = &val - &pi;
            neg ^= true;
        }

        debug_assert!(val <= pi);
        // Step 3.
        if val > pi_half {
            val = pi - val;
        }
        debug_assert!(val <= pi_half);

        let res = Self::sin_step4_reduction(&val, 16);
        let res = if neg { res.neg() } else { res };
        res.cast(orig_sem)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_sin_known_value() {
    use crate::std::string::ToString;
    // Verify the results with:
    // from mpmath import mp
    // mp.dps = 1000
    // mp.sin(801./10000)
    let res = Float::from_f64(801. / 10000.).sin().to_string();
    assert_eq!(res, ".08001437374006335");
    let res = Float::from_f64(90210. / 10000.).sin().to_string();
    assert_eq!(res, ".3928952872542333");
    let res = Float::from_f64(95051.).sin().to_string();
    assert_eq!(res, "-.8559198239971502");
}

#[cfg(feature = "std")]
#[test]
fn test_sin() {
    use super::utils;
    use super::FP128;

    for i in -100..100 {
        let f0 = i as f64;
        let r0 = f0.sin();
        let r1 = Float::from_f64(f0).cast(FP128).sin().as_f64();
        assert_eq!(r0, r1);
    }

    for i in -300..300 {
        let f0 = (i as f64) / 100.;
        let r0 = f0.sin();
        let r1 = Float::from_f64(f0).cast(FP128).sin().as_f64();
        assert_eq!(r0, r1);
    }

    // Test non-normal values.
    for v in utils::get_special_test_values() {
        if v.is_normal() {
            continue;
        }
        let r0 = v.sin();
        let r1 = Float::from_f64(v).sin().as_f64();
        assert_eq!(r0.is_nan(), r1.is_nan());
        if !r0.is_nan() {
            assert_eq!(r0, r1);
        }
    }
}

impl Float {
    /// Compute log(2).
    pub fn ln2(sem: Semantics) -> Self {
        // Represent log(2) using the sum 1/k*2^k
        let one = Self::one(sem, false);
        let mut sum = Self::zero(sem, false);
        let mut prev = Self::inf(sem, true);
        for k in 1..500 {
            let k2 = Self::from_u64(sem, 1).scale(k, RoundingMode::Zero);
            let k = Self::from_u64(sem, k as u64);
            let term = &one / &(k * k2);

            sum = &sum + &term;

            if prev == sum {
                break;
            }
            prev = sum.clone();
        }
        sum
    }

    /// Computes the taylor series, centered around 1, and valid in [0..2].
    /// z = (x - 1)/(x + 1)
    /// log(x) = 2 (z + z^3/3 + z^5/5 + z^7/7 ... )
    fn log_taylor(x: &Self) -> Self {
        let sem = x.get_semantics();
        let one = Self::one(sem, false);
        let z = &(x - &one) / &(x + &one);
        let z2 = z.sqr();

        let mut top = z;
        let mut sum = Self::zero(sem, false);
        let mut prev = Self::one(sem, true);
        for i in 0..50 {
            if prev == sum {
                break; // Stop if we are not making progress.
            }
            prev = sum.clone();

            let elem = &top / &Self::from_u64(sem, i * 2 + 1);
            sum += elem;

            // Prepare the next iteration.
            top = &top * &z2;
        }

        sum.scale(1, RoundingMode::Zero)
    }

    /// Reduce the range of 'x' with the identity:
    /// ln(x) = ln(sqrt(x)^2) = 2 * ln(sqrt(x)) and
    /// ln(x) = -ln(1/x)
    fn log_range_reduce(x: &Self) -> Self {
        let sem = x.get_semantics();
        let up = Self::from_f64(1.001).cast(sem);
        let one = Self::from_u64(sem, 1);

        if x > &up {
            let two = Self::from_u64(sem, 2);
            let sx = x.sqrt();
            return two * Self::log_range_reduce(&sx);
        }

        if x < &one {
            return Self::log_range_reduce(&(&one / x)).neg();
        }

        Self::log_taylor(x)
    }

    /// Computes logarithm of 'x'.
    pub fn log(&self) -> Self {
        let sem = self.get_semantics();

        //Fast Logarithm function for Arbitrary Precision number,
        // by Henrik Vestermark.

        // Handle all of the special cases:
        if !self.is_normal() || self.is_negative() {
            return Self::nan(sem, self.get_sign());
        }

        Self::log_range_reduce(self)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_ln2() {
    use super::FP128;
    assert_eq!(Float::ln2(FP128).as_f64(), std::f64::consts::LN_2);
}

#[test]
fn test_log() {
    use super::FP128;
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
            bottom = bottom * BigInt::from_u64(k);
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
        if !self.is_normal() {
            return Self::nan(sem, self.get_sign());
        }

        // Handle the negative values.
        if self.is_negative() {
            let one = Self::one(sem, false);
            return one / self.neg().exp();
        }

        Self::exp_range_reduce(self)
    }
}

#[test]
fn test_exp() {
    use super::FP128;
    assert_eq!(
        Float::from_f64(2.51).cast(FP128).exp().as_f64(),
        12.30493006051041
    );

    for x in [
        0.000003, 0.001, 0.12, 0.13, 0.5, 1.2, 2.3, 4.5, 9.8, 5.0, 11.2, 15.2,
        25.0, 34.001, 54., 89.1, 91.2, 102.2, 150., 192.4, 212., 256., 102.3,
    ] {
        let lhs = Float::from_f64(x).cast(FP128).exp().as_f64();
        let rhs = x.exp();
        assert_eq!(lhs, rhs);
    }
}

#[test]
fn test_powi() {
    assert_eq!(Float::from_f64(2.).powi(0).as_f64(), 1.);
    assert_eq!(Float::from_f64(2.).powi(1).as_f64(), 2.);
    assert_eq!(Float::from_f64(2.).powi(3).as_f64(), 8.);
    assert_eq!(Float::from_f64(2.).powi(5).as_f64(), 32.);
    assert_eq!(Float::from_f64(2.).powi(10).as_f64(), 1024.);
}
