use crate::float::Float;
use crate::RoundingMode;

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
            bottom *= next_term;
            neg ^= true;
        }

        sum
    }

    /// Reduce sin(x) in the range 0..pi/2, using the identity:
    /// sin(3x) = 3sin(x)-4(sin(x)^3)
    fn sin_step4_reduction(x: &Self, steps: usize) -> Self {
        use RoundingMode::None as rm;
        if steps == 0 {
            return Self::sin_taylor(x);
        }
        let i3 = Float::from_u64(x.get_semantics(), 3);
        let x3 = Float::div_with_rm(x, &i3, rm);
        let sx = Float::sin_step4_reduction(&x3, steps - 1);
        let sx3 = Float::mul_with_rm(&sx, &i3, rm);
        Float::sub_with_rm(&sx3, &sx.powi(3).scale(2, rm), rm)
    }

    /// Computes the sine of the number (in radians).
    pub fn sin(&self) -> Self {
        use RoundingMode::None as rm;
        // Fast Trigonometric functions for Arbitrary Precision number
        // by Henrik Vestermark.

        if self.is_zero() || self.is_nan() {
            return self.clone();
        }

        if self.is_inf() {
            return Self::nan(self.get_semantics(), self.get_sign());
        }

        let orig_sem = self.get_semantics();
        let sem = orig_sem.grow_log(12).increase_exponent(4);

        assert!(self.is_normal());

        let mut neg = false;

        let mut val = self.cast_with_rm(sem, rm);

        // Handle the negatives.
        if val.is_negative() {
            val = val.neg();
            neg ^= true;
        }

        // Range reductions.
        let is_small = self.get_exp() < 0;

        if !is_small {
            let pi = Self::pi(sem);
            let pi2 = pi.scale(1, rm);
            let pi_half = pi.scale(-1, rm);

            // Step 1
            if val > pi2 {
                val = val.rem(&pi2);
            }

            debug_assert!(val <= pi2);
            // Step 2.
            if val > pi {
                val = Float::sub_with_rm(&val, &pi, rm);
                neg ^= true;
            }

            debug_assert!(val <= pi);
            // Step 3.
            if val > pi_half {
                val = Float::sub_with_rm(&pi, &val, rm);
            }
            debug_assert!(val <= pi_half);
        }

        // Calculate the number of needed reduction: 8[2/3 * log(2) * log(p)];
        let k = orig_sem.log_precision() * 4;

        let res = Self::sin_step4_reduction(&val, k);
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
    use crate::utils;

    for i in -100..100 {
        let f0 = i as f64;
        let r0 = f0.sin();
        let r1 = Float::from_f64(f0).sin().as_f64();
        assert_eq!(r0, r1);
    }

    for i in -300..300 {
        let f0 = (i as f64) / 100.;
        let r0 = f0.sin();
        let r1 = Float::from_f64(f0).sin().as_f64();
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
    /// cos(x) = 1 - x^2 / 2! + x^4 / 4! - x^6/6! ....
    fn cos_taylor(x: &Self) -> Self {
        use crate::bigint::BigInt;
        let sem = x.get_semantics();

        let mut neg = false;
        let mut top = Self::one(sem, false);
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
            let next_term = BigInt::from_u64((i * 2 - 1) * (i * 2));
            bottom *= next_term;

            neg ^= true;
        }

        sum
    }

    /// Reduce cos(x) in the range 0..pi/2, using the identity:
    /// cos(2x) = 2cos(x)^2 - 1
    fn cos_step4_reduction(x: &Self, steps: usize) -> Self {
        use RoundingMode::None as rm;
        if steps == 0 {
            return Self::cos_taylor(x);
        }
        let sem = x.get_semantics();
        let one = Float::one(sem, false);
        let half_x = x.scale(-1, rm);
        let sx = Float::cos_step4_reduction(&half_x, steps - 1);
        Float::sub_with_rm(&sx.sqr().scale(1, rm), &one, rm)
    }

    /// Computes the cosine of the number (in radians).
    pub fn cos(&self) -> Self {
        use RoundingMode::None as rm;
        // Fast Trigonometric functions for Arbitrary Precision number
        // by Henrik Vestermark.

        if self.is_nan() {
            return self.clone();
        }

        if self.is_zero() {
            return Self::one(self.get_semantics(), false);
        }

        if self.is_inf() {
            return Self::nan(self.get_semantics(), self.get_sign());
        }

        let orig_sem = self.get_semantics();
        let sem = orig_sem.grow_log(14).increase_exponent(4);

        assert!(self.is_normal());

        let mut neg = false;

        let mut val = self.cast_with_rm(sem, rm);

        // Handle the negatives.
        if val.is_negative() {
            val = val.neg();
        }

        // Range reductions.
        let is_small = self.get_exp() < 0; // X < 1.

        if !is_small {
            let pi = Self::pi(sem);
            let pi2 = pi.scale(1, rm);
            let pi_half = pi.scale(-1, rm);

            // Step 1
            if val > pi2 {
                val = val.rem(&pi2);
            }
            debug_assert!(val <= pi2);

            // Step 2.
            if val > pi {
                val = Float::sub_with_rm(&pi2, &val, rm);
            }

            debug_assert!(val <= pi);
            // Step 3.
            if val > pi_half {
                val = Float::sub_with_rm(&pi, &val, rm);
                neg ^= true;
            }
            debug_assert!(val <= pi_half);
        }

        // Calculate the number of needed reduction: 2[log(2) * log(p)];
        let k = (sem.log_precision() * 8) / 10;

        let res = Self::cos_step4_reduction(&val, k);
        let res = if neg { res.neg() } else { res };
        res.cast(orig_sem)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_cos_known_value() {
    use crate::std::string::ToString;

    // Verify the results with:
    // from mpmath import mp
    // mp.dps = 100
    // mp.cos(801./10000)
    let res = Float::from_f64(801. / 10000.).cos().to_string();
    assert_eq!(res, ".9967937098492272");
    let res = Float::from_f64(2.3).cos().to_string();
    assert_eq!(res, "-.6662760212798241");
    let res = Float::from_f64(90210. / 10000.).cos().to_string();
    assert_eq!(res, "-.9195832171442742");
    let res = Float::from_f64(95051.).cos().to_string();
    assert_eq!(res, ".5171085523259959");
}

#[cfg(feature = "std")]
#[test]
fn test_cos() {
    use crate::utils;

    for i in -100..100 {
        let f0 = i as f64;
        let r0 = f0.cos();
        let r1 = Float::from_f64(f0).cos().as_f64();
        assert_eq!(r0, r1);
    }

    // The native implementation of sin is not accurate to all 64 bits, so
    // we just pick a few values where we happen to get lucky and native sin
    // matches the arbitrary precision implementation.
    for i in -100..100 {
        let f0 = (i as f64) / 100.;
        let r0 = f0.cos();
        let r1 = Float::from_f64(f0).cos().as_f64();
        assert_eq!(r0, r1);
    }

    // Test non-normal values.
    for v in utils::get_special_test_values() {
        if v.is_normal() {
            continue;
        }
        let r0 = v.cos();
        let r1 = Float::from_f64(v).cos().as_f64();
        assert_eq!(r0.is_nan(), r1.is_nan());
        if !r0.is_nan() {
            assert_eq!(r0, r1);
        }
    }
}

impl Float {
    /// Computes the tangent of the number (in radians).
    pub fn tan(&self) -> Self {
        use RoundingMode::None as rm;
        // Fast Trigonometric functions for Arbitrary Precision number
        // by Henrik Vestermark.

        if self.is_zero() || self.is_nan() {
            return self.clone();
        }

        if self.is_inf() {
            return Self::nan(self.get_semantics(), self.get_sign());
        }

        let orig_sem = self.get_semantics();
        let sem = orig_sem.grow_log(12).increase_exponent(4);

        assert!(self.is_normal());

        let mut neg = false;

        let mut val = self.cast_with_rm(sem, rm);

        // Handle the negatives.
        if val.is_negative() {
            val = val.neg();
            neg ^= true;
        }

        // Range reductions.
        let is_small = self.get_exp() < 0;

        if !is_small {
            let pi = Self::pi(sem);
            let half_pi = pi.scale(-1, rm);

            // Wrap around pi.
            if val > pi {
                val = val.rem(&pi);
            }
            debug_assert!(val <= pi);

            // Reduce to 0..pi/2.
            if val > half_pi {
                val = pi - val;
                neg ^= true;
            }
            debug_assert!(val <= half_pi);
        }

        // Tan(x) = sin(x)/sqrt(1-sin(x)^2).
        let sinx = val.sin();
        let one = Float::one(sem, false);
        let bottom = (one - sinx.sqr()).sqrt();
        let res = sinx / bottom;
        let res = if neg { res.neg() } else { res };
        res.cast(orig_sem)
    }
}

#[cfg(feature = "std")]
#[test]
fn test_tan_known_value() {
    use crate::std::string::ToString;

    // Verify the results with:
    // from mpmath import mp
    // mp.dps = 100
    // mp.tan(801./10000)
    let res = Float::from_f64(801. / 10000.).tan().to_string();
    assert_eq!(res, ".08027174825588148");
    let res = Float::from_f64(2.3).tan().to_string();
    assert_eq!(res, "-1.1192136417341325");
    let res = Float::from_f64(90210. / 10000.).tan().to_string();
    assert_eq!(res, "-.4272536513599634");
    let res = Float::from_f64(95051.).tan().to_string();
    assert_eq!(res, "-1.6552033806966715");
}
