use super::bigint::LossFraction;
use std::ops::{Add, Div, Mul, Sub};

use super::float::{
    shift_right_with_loss, Category, Float, MantissaTy, RoundingMode,
};

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// An inner function that performs the addition and subtraction of normal
    /// numbers (no NaN, Inf, Zeros).
    /// See Pg 247.  Chapter 8. Algorithms for the Five Basic Operations.
    /// This implementation follows the APFloat implementation, that does not
    /// swap the operands.
    fn add_or_sub_normals(
        mut a: Self,
        mut b: Self,
        subtract: bool,
    ) -> (Self, LossFraction) {
        let loss;

        // Align the input numbers on the same exponent.
        let bits = a.get_exp() - b.get_exp();

        // Can transform (a-b) to (a + -b), either way, there are cases where
        // subtraction needs to happen.
        let subtract = subtract ^ (a.get_sign() ^ b.get_sign());
        if subtract {
            // Align the input numbers. We shift LHS one bit to the left to
            // allow carry/borrow in case of underflow as result of subtraction.
            match bits.cmp(&0) {
                std::cmp::Ordering::Equal => {
                    loss = LossFraction::ExactlyZero;
                }
                std::cmp::Ordering::Greater => {
                    loss = b.shift_significand_right((bits - 1) as u64);
                    a.shift_significand_left(1);
                }
                std::cmp::Ordering::Less => {
                    loss = a.shift_significand_right((-bits - 1) as u64);
                    b.shift_significand_left(1);
                }
            }

            let a_mantissa = a.get_mantissa();
            let b_mantissa = b.get_mantissa();
            let ab_mantissa;
            let mut sign = a.get_sign();

            // Figure out the carry from the shifting operations that dropped
            // bits.
            let c = !loss.is_exactly_zero() as u64;
            let c = MantissaTy::from_u64(c);

            // Figure out which mantissa is larger, to make sure that we don't
            // overflow the subtraction.
            if a_mantissa < b_mantissa {
                // A < B
                ab_mantissa = b_mantissa - a_mantissa - c;
                sign = !sign;
            } else {
                // A >= B
                ab_mantissa = a_mantissa - b_mantissa - c;
            }
            (Self::new(sign, a.get_exp(), ab_mantissa), loss.invert())
        } else {
            // Handle the easy case of Add:
            let mut b = b;
            let mut a = a;
            if bits > 0 {
                loss = b.shift_significand_right(bits as u64);
            } else {
                loss = a.shift_significand_right(-bits as u64);
            }
            debug_assert!(a.get_exp() == b.get_exp());
            let ab_mantissa = a.get_mantissa() + b.get_mantissa();
            (Self::new(a.get_sign(), a.get_exp(), ab_mantissa), loss)
        }
    }

    /// Computes a+b using the rounding mode `rm`.
    pub fn add_with_rm(a: Self, b: Self, rm: RoundingMode) -> Self {
        Self::add_sub(a, b, false, rm)
    }
    /// Computes a-b using the rounding mode `rm`.
    pub fn sub_with_rm(a: Self, b: Self, rm: RoundingMode) -> Self {
        Self::add_sub(a, b, true, rm)
    }

    fn add_sub(a: Self, b: Self, subtract: bool, rm: RoundingMode) -> Self {
        // Table 8.2: Specification of addition for positive floating-point
        // data. Pg 247.
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
                res.0.normalize(rm, res.1);
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
    use crate::utils;

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
            assert_eq!(r0.is_normal(), r1.is_normal());
            // Check that the results are bit identical, or are both NaN.
            assert!(!r0.is_normal() || r0_bits == r1_bits);
        }
    }
}

#[test]
fn test_add_random_vals() {
    use crate::utils;
    use crate::FP64;

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
    /// Compute a*b using the rounding mode `rm`.
    pub fn mul_with_rm(a: Self, b: Self, rm: RoundingMode) -> Self {
        let sign = a.get_sign() ^ b.get_sign();

        // Table 8.4: Specification of multiplication for floating-point data of
        // positive sign. Page 251.
        match (a.get_category(), b.get_category()) {
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
                res.normalize(rm, loss);
                res
            }
        }
    }

    /// See Pg 251. 8.4 Floating-Point Multiplication
    fn mul_normals(a: Self, b: Self, sign: bool) -> (Self, LossFraction) {
        // We multiply digits in the format 1.xx * 2^(e), or mantissa * 2^(e+1).
        // When we multiply two 2^(e+1) numbers, we get:
        // log(2^(e_a+1)*2^(e_b+1)) = e_a + e_b + 2.
        let mut exp = a.get_exp() + b.get_exp();

        let mut loss = LossFraction::ExactlyZero;

        let a_significand = a.get_mantissa();
        let b_significand = b.get_mantissa();

        let mut ab_significand = a_significand * b_significand;
        let first_non_zero = ab_significand.msb_index() as u64;

        // The exponent is correct, but the bits are not in the right place.
        // Set the right exponent for where the bits are placed, and fix the
        // exponent below.
        exp -= MANTISSA as i64;

        let precision = Self::get_precision();
        if first_non_zero > precision {
            let bits = first_non_zero - precision;

            (ab_significand, loss) =
                shift_right_with_loss(ab_significand, bits);
            exp += bits as i64;
        }

        (Self::new(sign, exp, ab_significand), loss)
    }
}

#[test]
fn test_mul_simple() {
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
    use super::utils;

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
    use super::utils;
    use crate::FP64;
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

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Compute a/b, with the rounding mode `rm`.
    pub fn div_with_rm(a: Self, b: Self, rm: RoundingMode) -> Self {
        let sign = a.get_sign() ^ b.get_sign();
        // Table 8.5: Special values for x/y - Page 263.
        match (a.get_category(), b.get_category()) {
            (Category::NaN, _)
            | (_, Category::NaN)
            | (Category::Zero, Category::Zero)
            | (Category::Infinity, Category::Infinity) => Self::nan(sign),

            (_, Category::Infinity) => Self::zero(sign),
            (Category::Zero, _) => Self::zero(sign),
            (_, Category::Zero) => Self::inf(sign),
            (Category::Infinity, _) => Self::inf(sign),
            (Category::Normal, Category::Normal) => {
                let (mut res, loss) = Self::div_normals(a, b);
                res.normalize(rm, loss);
                res
            }
        }
    }

    /// Compute a/b, where both `a` and `b` are normals.
    /// Page 262 8.6. Floating-Point Division.
    /// This implementation uses a regular integer division for the mantissa.
    fn div_normals(mut a: Self, mut b: Self) -> (Self, LossFraction) {
        // Start by normalizing the dividend and divisor to the MSB.
        a.align_mantissa(); // Normalize the dividend.
        b.align_mantissa(); // Normalize the divisor.

        let mut a_mantissa = a.get_mantissa();
        let b_mantissa = b.get_mantissa();

        // Calculate the sign and exponent.
        let mut exp = a.get_exp() - b.get_exp();
        let sign = a.get_sign() ^ b.get_sign();

        // Make sure that A >= B, to allow the integer division to generate all
        // of the bits of the result.
        if a_mantissa < b_mantissa {
            a_mantissa.shift_left(1);
            exp -= 1;
        }

        // The bits are now aligned to the MSB of the mantissa. The
        // semantics need to be 1.xxxxx, but we perform integer division.
        // Shift the dividend to make sure that we generate the bits after
        // the period.
        a_mantissa.shift_left(MANTISSA);
        let reminder = a_mantissa.inplace_div(b_mantissa);

        // Find 2 x reminder, to be able to compare to the reminder and figure
        // out the kind of loss that we have.
        let mut reminder_2x = reminder;
        reminder_2x.shift_left(1);

        let reminder = reminder_2x.cmp(&b_mantissa);
        let is_zero = reminder_2x.is_zero();
        let loss = match reminder {
            std::cmp::Ordering::Less => {
                if is_zero {
                    LossFraction::ExactlyZero
                } else {
                    LossFraction::LessThanHalf
                }
            }
            std::cmp::Ordering::Equal => LossFraction::ExactlyHalf,
            std::cmp::Ordering::Greater => LossFraction::MoreThanHalf,
        };

        let x = Self::new(sign, exp, a_mantissa);
        (x, loss)
    }
}

#[test]
fn test_div_simple() {
    use super::float::FP64;

    let a: f64 = 1.0;
    let b: f64 = 7.0;

    let af = FP64::from_f64(a);
    let bf = FP64::from_f64(b);
    let cf = FP64::div_with_rm(af, bf, RoundingMode::NearestTiesToEven);

    let r0 = cf.as_f64();
    let r1: f64 = a / b;
    assert_eq!(r0, r1);
}

#[test]
fn test_div_special_values() {
    use super::utils;

    // Test the multiplication of various irregular values.
    let values = utils::get_special_test_values();

    use super::float::FP64;

    fn div_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        FP64::div_with_rm(a, b, RoundingMode::NearestTiesToEven).as_f64()
    }

    for v0 in values {
        for v1 in values {
            let r0 = div_f64(v0, v1);
            let r1 = v0 / v1;
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

impl<const EXPONENT: usize, const MANTISSA: usize> Add
    for Float<EXPONENT, MANTISSA>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::add_with_rm(self, rhs, RoundingMode::NearestTiesToEven)
    }
}

impl<const EXPONENT: usize, const MANTISSA: usize> Sub
    for Float<EXPONENT, MANTISSA>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::sub_with_rm(self, rhs, RoundingMode::NearestTiesToEven)
    }
}

impl<const EXPONENT: usize, const MANTISSA: usize> Mul
    for Float<EXPONENT, MANTISSA>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::mul_with_rm(self, rhs, RoundingMode::NearestTiesToEven)
    }
}

impl<const EXPONENT: usize, const MANTISSA: usize> Div
    for Float<EXPONENT, MANTISSA>
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::div_with_rm(self, rhs, RoundingMode::NearestTiesToEven)
    }
}

#[test]
fn test_operators() {
    use crate::FP64;
    let a = FP64::from_f32(8.0);
    let b = FP64::from_f32(2.0);
    let c = a + b;
    let d = a - b;
    let e = a * b;
    let f = a / b;
    assert_eq!(c.as_f64(), 10.0);
    assert_eq!(d.as_f64(), 6.0);
    assert_eq!(e.as_f64(), 16.0);
    assert_eq!(f.as_f64(), 4.0);
}

#[test]
fn test_slow_sqrt_2_test() {
    use crate::FP128;
    use crate::FP64;

    // Find sqrt using a binary search.
    let two = FP128::from_f64(2.0);
    let mut high = FP128::from_f64(2.0);
    let mut low = FP128::from_f64(1.0);

    for _ in 0..25 {
        let mid = (high + low) / two;
        if (mid * mid) < two {
            low = mid;
        } else {
            high = mid;
        }
    }

    let res: FP64 = low.cast();
    assert!(res.as_f64() < 1.4142137_f64);
    assert!(res.as_f64() > 1.4142134_f64);
}

#[test]
fn test_famous_pentium4_bug() {
    // https://en.wikipedia.org/wiki/Pentium_FDIV_bug
    use crate::FP128;

    let a = FP128::from_u64(4_195_835);
    let b = FP128::from_u64(3_145_727);
    let res = a / b;
    let result = res.to_string();
    assert!(result.starts_with("1.333820449136241002"));
}
