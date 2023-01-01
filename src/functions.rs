use super::float::Float;

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Calculate the square root of the number using the Newton Raphson
    // method.
    pub fn sqrt(&self) -> Self {
        if self.is_zero() {
            return *self; // (+/-) zero
        } else if self.is_nan() || self.is_negative() {
            return Self::nan(self.get_sign()); // (-/+)Nan, -Number.
        } else if self.is_inf() {
            return *self; // Inf+.
        }

        let target = *self;
        let two = Self::from_u64(2);

        // Start the search at max(2, x).
        let mut x = if target < two { two } else { target };
        let mut prev = x;

        loop {
            x = (x + (target / x)) / two;
            // Stop when value did not change or regressed.
            if prev < x || x == prev {
                return x;
            }
            prev = x;
        }
    }

    /// Returns the absolute value of this float.
    pub fn abs(&self) -> Self {
        let mut x = *self;
        x.set_sign(false);
        x
    }

    /// Returns the greater of self and `other`.
    pub fn max(&self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        } else if other.is_nan() {
            return *self;
        } else if self.get_sign() != other.get_sign() {
            return if self.get_sign() { other } else { *self }; // Handle (+-)0.
        }
        if *self > other {
            *self
        } else {
            other
        }
    }

    /// Returns the smaller of self and `other`.
    pub fn min(&self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        } else if other.is_nan() {
            return *self;
        } else if self.get_sign() != other.get_sign() {
            return if self.get_sign() { *self } else { other }; // Handle (+-)0.
        }
        if *self > other {
            other
        } else {
            *self
        }
    }
}

#[test]
fn test_sqrt() {
    use super::utils;
    use super::FP64;

    // Try a few power-of-two values.
    for i in 0..256 {
        let v16 = FP64::from_u64(i * i);
        assert_eq!(v16.sqrt().as_f64(), (i) as f64);
    }

    // Test the category and value of the different special values (inf, zero,
    // correct sign, etc).
    for v_f64 in utils::get_special_test_values() {
        let vf = FP64::from_f64(v_f64);
        assert_eq!(vf.sqrt().is_inf(), v_f64.sqrt().is_infinite());
        assert_eq!(vf.sqrt().is_nan(), v_f64.sqrt().is_nan());
        assert_eq!(vf.sqrt().is_negative(), v_f64.sqrt().is_sign_negative());
    }

    // Test precomputed values.
    fn check(inp: f64, res: f64) {
        assert_eq!(FP64::from_f64(inp).sqrt().as_f64(), res);
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

#[test]
fn test_min_max() {
    use super::utils;
    use super::FP64;

    fn check(v0: f64, v1: f64) {
        // Min.
        let correct = v0.min(v1);
        let test = FP64::from_f64(v0).min(FP64::from_f64(v1)).as_f64();
        assert_eq!(test.is_nan(), correct.is_nan());
        if !correct.is_nan() {
            assert_eq!(correct, test);
        }
        // Max.
        let correct = v0.max(v1);
        let test = FP64::from_f64(v0).max(FP64::from_f64(v1)).as_f64();
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

#[test]
fn test_abs() {
    use super::utils;
    use super::FP64;
    for v in utils::get_special_test_values() {
        if !v.is_nan() {
            assert_eq!(FP64::from_f64(v).abs().as_f64(), v.abs());
        }
    }
}
