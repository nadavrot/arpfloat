use super::float::{Category, Float, LossFraction, RoundingMode};
use super::utils;

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    pub fn add_or_sub_normals(
        a: Self,
        b: Self,
        subtract: bool,
    ) -> (Self, LossFraction) {
        let loss;

        // Align the input numbers on the same exponent.
        let bits = a.get_exp() - b.get_exp();

        // Can transform (a-b) to (a + -b), either way, there are cases where
        // subtraction needs to happen.
        let subtract = subtract ^ (a.get_sign() ^ b.get_sign());
        if subtract {
            let mut b = b;
            let mut a = a;

            // Align the input numbers. We shift LHS one bit to the left to
            // allow carry/borrow in case of underflow as result of subtraction.
            if bits == 0 {
                loss = LossFraction::ExactlyZero;
            } else if bits > 0 {
                loss = b.shift_significand_right((bits - 1) as u64);
                a.shift_significand_left(1);
            } else {
                loss = a.shift_significand_right((-bits - 1) as u64);
                b.shift_significand_left(1);
            }

            let a_mantissa = a.get_mantissa();
            let b_mantissa = b.get_mantissa();
            let ab_mantissa;
            let mut sign = a.get_sign();

            // Figure out the carry from the shifting operations that dropped
            // bits.
            let c = if let LossFraction::ExactlyZero = loss {
                0
            } else {
                1
            };

            // Figure out which mantissa is larger, to make sure that we don't
            // overflow the subtraction.
            if a.absolute_less_than(b) {
                // A < B
                ab_mantissa = b_mantissa - a_mantissa - c;
                sign = !sign;
            } else {
                // A >= B
                ab_mantissa = a_mantissa - b_mantissa - c;
            }
            return (Self::new(sign, a.get_exp(), ab_mantissa), loss.invert());
        } else {
            // Handle the easy case of Add:
            let mut b = b;
            let mut a = a;
            let ab_mantissa;
            if bits > 0 {
                loss = b.shift_significand_right((bits) as u64);
            } else {
                loss = a.shift_significand_right((-bits) as u64);
            }
            assert!(a.get_exp() == b.get_exp());
            ab_mantissa = a.get_mantissa() + b.get_mantissa();
            return (Self::new(a.get_sign(), a.get_exp(), ab_mantissa), loss);
        }
    }

    pub fn add(a: Self, b: Self) -> Self {
        Self::add_sub(a, b, false)
    }

    pub fn sub(a: Self, b: Self) -> Self {
        Self::add_sub(a, b, true)
    }

    pub fn add_sub(a: Self, b: Self, subtract: bool) -> Self {
        match (a.get_category(), b.get_category()) {
            (Category::NaN, Category::Infinity)
            | (Category::NaN, Category::NaN)
            | (Category::NaN, Category::Normal)
            | (Category::NaN, Category::Zero)
            | (Category::Normal, Category::Zero)
            | (Category::Infinity, Category::Normal)
            | (Category::Infinity, Category::Zero) => a,

            (Category::Zero, Category::NaN)
            | (Category::Normal, Category::NaN)
            | (Category::Infinity, Category::NaN) => Self::nan(b.get_sign()),

            (Category::Normal, Category::Infinity)
            | (Category::Zero, Category::Infinity) => {
                Self::inf(b.get_sign() ^ subtract)
            }

            (Category::Zero, Category::Normal) => b,

            (Category::Zero, Category::Zero) => {
                Self::zero(a.get_sign() && b.get_sign())
            }

            (Category::Infinity, Category::Infinity) => {
                if a.get_sign() ^ b.get_sign() ^ subtract {
                    return Self::nan(a.get_sign() ^ b.get_sign());
                }
                Self::inf(a.get_sign())
            }

            (Category::Normal, Category::Normal) => {
                let mut res = Self::add_or_sub_normals(a, b, subtract);
                res.0.normalize(RoundingMode::NearestTiesToEven, res.1);
                res.0
            }
        }
    }
}

#[test]
fn test_add() {
    use super::float::FP64;
    let a = FP64::from_u64(1);
    let b = FP64::from_u64(2);
    let _ = FP64::add(a, b);
}

#[test]
fn test_addition() {
    use super::float::FP64;

    fn add_helper(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        let c = FP64::add(a, b);
        c.as_f64()
    }

    assert_eq!(add_helper(0., -4.), -4.);
    assert_eq!(add_helper(-4., 0.), -4.);
    assert_eq!(add_helper(1., 1.), 2.);
    assert_eq!(add_helper(8., 4.), 12.);
    assert_eq!(add_helper(8., 4.), 12.);
    assert_eq!(add_helper(128., 2.), 130.);
    assert_eq!(add_helper(128., -8.), 120.);
    assert_eq!(add_helper(64., -60.), 4.);
    assert_eq!(add_helper(69., -65.), 4.);
    assert_eq!(add_helper(69., 69.), 138.);
    assert_eq!(add_helper(69., 1.), 70.);
    assert_eq!(add_helper(-128., -8.), -136.);
    assert_eq!(add_helper(64., -65.), -1.);
    assert_eq!(add_helper(-64., -65.), -129.);
    assert_eq!(add_helper(-15., -15.), -30.);

    assert_eq!(add_helper(-15., 15.), 0.);

    for i in -4..15 {
        for j in i..15 {
            assert_eq!(
                add_helper(f64::from(j), f64::from(i)),
                f64::from(i) + f64::from(j)
            );
        }
    }
}

// Pg 120.  Chapter 4. Basic Properties and Algorithms.
#[test]
fn test_addition_large_numbers() {
    use super::float::FP64;

    let one = FP64::from_i64(1);
    let mut a = FP64::from_i64(1);

    while FP64::sub(FP64::add(a, one), a) == one {
        a = FP64::add(a, a);
    }

    let mut b = one;
    while FP64::sub(FP64::add(a, b), a) != b {
        b = FP64::add(b, one);
    }

    assert_eq!(a.as_f64(), 9007199254740992.);
    assert_eq!(b.as_f64(), 2.);
}

#[test]
fn add_denormals() {
    use super::float::FP64;

    let v0 = f64::from_bits(0x0000_0000_0010_0010);
    let v1 = f64::from_bits(0x0000_0000_1001_0010);
    let v2 = f64::from_bits(0x1000_0000_0001_0010);
    assert_eq!(add_f64(v2, -v1), v2 - v1);

    let a0 = FP64::from_f64(v0);
    assert_eq!(a0.as_f64(), v0);

    fn add_f64(a: f64, b: f64) -> f64 {
        let a0 = FP64::from_f64(a);
        let b0 = FP64::from_f64(b);
        assert_eq!(a0.as_f64(), a);
        FP64::add(a0, b0).as_f64()
    }

    // Add and subtract denormals.
    assert_eq!(add_f64(v0, v1), v0 + v1);
    assert_eq!(add_f64(v0, -v0), v0 - v0);
    assert_eq!(add_f64(v0, v2), v0 + v2);
    assert_eq!(add_f64(v2, v1), v2 + v1);
    assert_eq!(add_f64(v2, -v1), v2 - v1);

    // Add and subtract denormals and normal numbers.
    assert_eq!(add_f64(v0, 10.), v0 + 10.);
    assert_eq!(add_f64(v0, -10.), v0 - 10.);
    assert_eq!(add_f64(10000., v0), 10000. + v0);
}

#[test]
fn add_special_values() {
    use crate::base::utils;

    // Test the addition of various irregular values.
    let values = utils::get_special_test_values();

    use super::float::FP64;

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        FP64::add(a, b).as_f64()
    }

    for v0 in values {
        for v1 in values {
            let r0 = add_f64(v0, v1);
            let r1 = v0 + v1;
            let r0_bits = r0.to_bits();
            let r1_bits = r1.to_bits();
            assert_eq!(r0.is_finite(), r1.is_finite());
            assert_eq!(r0.is_nan(), r1.is_nan());
            assert_eq!(r0.is_infinite(), r1.is_infinite());
            // Check that the results are bit identical, or are both NaN.
            assert!(!r0.is_nan() || r0_bits == r1_bits);
        }
    }
}

#[test]
fn test_add_random_vals() {
    use crate::base::utils;
    use crate::base::FP64;

    let mut lfsr = utils::Lfsr::new();

    let v0: u64 = 0x645e91f69778bad3;
    let v1: u64 = 0xe4d91b16be9ae0c5;

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        let k = FP64::add(a, b);
        k.as_f64()
    }

    let f0 = f64::from_bits(v0);
    let f1 = f64::from_bits(v1);

    let r0 = add_f64(f0, f1);
    let r1 = f0 + f1;

    assert_eq!(r0.is_finite(), r1.is_finite());
    assert_eq!(r0.is_nan(), r1.is_nan());
    assert_eq!(r0.is_infinite(), r1.is_infinite());
    let r0_bits = r0.to_bits();
    let r1_bits = r1.to_bits();
    // Check that the results are bit identical, or are both NaN.
    assert!(r1.is_nan() || r0_bits == r1_bits);

    for _ in 0..50000 {
        let v0 = lfsr.get64();
        let v1 = lfsr.get64();

        let f0 = f64::from_bits(v0);
        let f1 = f64::from_bits(v1);

        let r0 = add_f64(f0, f1);
        let r1 = f0 + f1;

        assert_eq!(r0.is_finite(), r1.is_finite());
        assert_eq!(r0.is_nan(), r1.is_nan());
        assert_eq!(r0.is_infinite(), r1.is_infinite());
        let r0_bits = r0.to_bits();
        let r1_bits = r1.to_bits();
        // Check that the results are bit identical, or are both NaN.
        assert!(r1.is_nan() || r0_bits == r1_bits);
    }
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    pub fn mul(a: Self, b: Self) -> Self {
        let sign = a.get_sign() ^ b.get_sign();

        return match (a.get_category(), b.get_category()) {
            (Category::Zero, Category::NaN)
            | (Category::Normal, Category::NaN)
            | (Category::Infinity, Category::NaN) => Self::nan(b.get_sign()),
            (Category::NaN, Category::Infinity)
            | (Category::NaN, Category::NaN)
            | (Category::NaN, Category::Normal)
            | (Category::NaN, Category::Zero) => Self::nan(a.get_sign()),
            (Category::Normal, Category::Infinity)
            | (Category::Infinity, Category::Normal)
            | (Category::Infinity, Category::Infinity) => Self::inf(sign),
            (Category::Normal, Category::Zero)
            | (Category::Zero, Category::Normal)
            | (Category::Zero, Category::Zero) => Self::zero(sign),

            (Category::Zero, Category::Infinity)
            | (Category::Infinity, Category::Zero) => Self::nan(sign),

            (Category::Normal, Category::Normal) => {
                let (mut res, loss) = Self::mul_normals(a, b, sign);
                res.normalize(RoundingMode::NearestTiesToEven, loss);
                res
            }
        };
    }

    fn shift_right_with_loss128(val: u128, bits: u64) -> (u128, LossFraction) {
        // See "get_loss_kind_of_trunc".
        fn get_loss_kind_of_trunc128(val: u128) -> LossFraction {
            if val == 0 {
                return LossFraction::ExactlyZero;
            } else if val == (1 << 127) {
                return LossFraction::ExactlyHalf;
            } else if val > (1 << 127) {
                return LossFraction::MoreThanHalf;
            }
            LossFraction::LessThanHalf
        }

        if bits == 0 {
            (val, LossFraction::ExactlyZero)
        } else if bits < 128 {
            let loss = get_loss_kind_of_trunc128(val << (128 - bits));
            (val >> bits, loss)
        } else {
            let loss = get_loss_kind_of_trunc128(val);
            (0, loss)
        }
    }

    pub fn mul_normals(a: Self, b: Self, sign: bool) -> (Self, LossFraction) {
        // We multiply digits in the format 1.xx * 2^(e), or mantissa * 2^(e+1).
        // When we multiply two 2^(e+1) numbers, we get:
        // log(2^(e_a+1)*2^(e_b+1)) = e_a + e+b + 2.
        let mut exp = a.get_exp() + b.get_exp();

        let mut loss = LossFraction::ExactlyZero;

        let a_significand = a.get_mantissa() as u128;
        let b_significand = b.get_mantissa() as u128;
        let mut ab_significand = a_significand * b_significand;
        let first_non_zero = utils::next_msb128(ab_significand);

        // The exponent is correct, but the bits are not in the right place.
        // Set the right exponent for where the bits are placed, and fix the
        // exponent below.
        exp -= MANTISSA as i64;

        let precision = Self::get_precision();
        if first_non_zero > precision {
            let bits = first_non_zero - precision;

            (ab_significand, loss) =
                Self::shift_right_with_loss128(ab_significand, bits);
            exp += bits as i64;
        }

        (Self::new(sign, exp, ab_significand as u64), loss)
    }
}

#[test]
fn test_simple() {
    use super::float::FP64;

    let a: f64 = -24.0;
    let b: f64 = 0.1;

    let af = FP64::from_f64(a);
    let bf = FP64::from_f64(b);
    let cf = FP64::mul(af, bf);

    let r0 = cf.as_f64();
    let r1: f64 = a * b;
    assert_eq!(r0, r1);
}

#[test]
fn mul_regular_values() {
    // Test the addition of regular values.
    let values = [-5.0, 0., -0., 24., 1., 11., 10000., 256., 0.1, 3., 17.5];
    use super::float::FP64;

    fn mul_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        FP64::mul(a, b).as_f64()
    }

    for v0 in values {
        for v1 in values {
            let r0 = mul_f64(v0, v1);
            let r1 = v0 * v1;
            let r0_bits = r0.to_bits();
            let r1_bits = r1.to_bits();
            // Check that the results are bit identical, or are both NaN.
            assert_eq!(r0_bits, r1_bits);
        }
    }
}

#[test]
fn test_mul_special_values() {
    // Test the multiplication of various irregular values.
    let values = utils::get_special_test_values();

    use super::float::FP64;

    fn mul_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        FP64::mul(a, b).as_f64()
    }

    for v0 in values {
        for v1 in values {
            let r0 = mul_f64(v0, v1);
            let r1 = v0 * v1;
            assert_eq!(r0.is_finite(), r1.is_finite());
            assert_eq!(r0.is_nan(), r1.is_nan());
            assert_eq!(r0.is_infinite(), r1.is_infinite());
            let r0_bits = r0.to_bits();
            let r1_bits = r1.to_bits();
            // Check that the results are bit identical, or are both NaN.
            assert!(r1.is_nan() || r0_bits == r1_bits);
        }
    }
}

#[test]
fn test_mul_random_vals() {
    use crate::base::FP64;
    let mut lfsr = utils::Lfsr::new();

    fn mul_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        let k = FP64::mul(a, b);
        k.as_f64()
    }

    for _ in 0..50000 {
        let v0 = lfsr.get64();
        let v1 = lfsr.get64();

        let f0 = f64::from_bits(v0);
        let f1 = f64::from_bits(v1);

        let r0 = mul_f64(f0, f1);
        let r1 = f0 * f1;
        assert_eq!(r0.is_finite(), r1.is_finite());
        assert_eq!(r0.is_nan(), r1.is_nan());
        assert_eq!(r0.is_infinite(), r1.is_infinite());
        let r0_bits = r0.to_bits();
        let r1_bits = r1.to_bits();
        // Check that the results are bit identical, or are both NaN.
        assert!(r1.is_nan() || r0_bits == r1_bits);
    }
}