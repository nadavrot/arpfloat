use crate::{RoundingMode};

use crate::float::Float;

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

    /// Return the sine function.
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
        let sem = orig_sem.grow_log(14).increase_exponent(4);

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
