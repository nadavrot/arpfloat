//! This module contains the implementation of the basic arithmetic operations:
//! Addition, Subtraction, Multiplication, Division.
extern crate alloc;
use crate::bigint::BigInt;

use super::bigint::LossFraction;
use super::float::{Category, Float, RoundingMode};
use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign,
};

impl Float {
    /// An inner function that performs the addition and subtraction of normal
    /// numbers (no NaN, Inf, Zeros).
    /// See Pg 247.  Chapter 8. Algorithms for the Five Basic Operations.
    /// This implementation follows the APFloat implementation, that does not
    /// swap the operands.
    fn add_or_sub_normals(
        a: &Self,
        b: &Self,
        subtract: bool,
    ) -> (Self, LossFraction) {
        debug_assert_eq!(a.get_semantics(), b.get_semantics());
        let sem = a.get_semantics();
        let loss;
        let mut a = a.clone();
        let mut b = b.clone();

        // Align the input numbers on the same exponent.
        let bits = a.get_exp() - b.get_exp();

        // Can transform (a-b) to (a + -b), either way, there are cases where
        // subtraction needs to happen.
        let subtract = subtract ^ (a.get_sign() ^ b.get_sign());
        if subtract {
            // Align the input numbers. We shift LHS one bit to the left to
            // allow carry/borrow in case of underflow as result of subtraction.
            match bits.cmp(&0) {
                Ordering::Equal => {
                    loss = LossFraction::ExactlyZero;
                }
                Ordering::Greater => {
                    loss = b.shift_significand_right((bits - 1) as u64);
                    a.shift_significand_left(1);
                }
                Ordering::Less => {
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
            let c = BigInt::from_u64(c);

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
            (
                Self::from_parts(sem, sign, a.get_exp(), ab_mantissa),
                loss.invert(),
            )
        } else {
            // Handle the easy case of Add:
            let mut b = b.clone();
            let mut a = a.clone();
            if bits > 0 {
                loss = b.shift_significand_right(bits as u64);
            } else {
                loss = a.shift_significand_right(-bits as u64);
            }
            debug_assert!(a.get_exp() == b.get_exp());
            let ab_mantissa = a.get_mantissa() + b.get_mantissa();
            (
                Self::from_parts(sem, a.get_sign(), a.get_exp(), ab_mantissa),
                loss,
            )
        }
    }

    /// Computes a+b using the rounding mode `rm`.
    pub fn add_with_rm(a: &Self, b: &Self, rm: RoundingMode) -> Self {
        Self::add_sub(a, b, false, rm)
    }
    /// Computes a-b using the rounding mode `rm`.
    pub fn sub_with_rm(a: &Self, b: &Self, rm: RoundingMode) -> Self {
        Self::add_sub(a, b, true, rm)
    }

    fn add_sub(a: &Self, b: &Self, subtract: bool, rm: RoundingMode) -> Self {
        let sem = a.get_semantics();
        // Table 8.2: Specification of addition for positive floating-point
        // data. Pg 247.
        match (a.get_category(), b.get_category()) {
            (Category::NaN, Category::Infinity)
            | (Category::NaN, Category::NaN)
            | (Category::NaN, Category::Normal)
            | (Category::NaN, Category::Zero)
            | (Category::Normal, Category::Zero)
            | (Category::Infinity, Category::Normal)
            | (Category::Infinity, Category::Zero) => a.clone(),

            (Category::Zero, Category::NaN)
            | (Category::Normal, Category::NaN)
            | (Category::Infinity, Category::NaN) => {
                Self::nan(sem, b.get_sign())
            }

            (Category::Normal, Category::Infinity)
            | (Category::Zero, Category::Infinity) => {
                Self::inf(sem, b.get_sign() ^ subtract)
            }

            (Category::Zero, Category::Normal) => Self::from_parts(
                sem,
                b.get_sign() ^ subtract,
                b.get_exp(),
                b.get_mantissa(),
            ),

            (Category::Zero, Category::Zero) => {
                Self::zero(sem, a.get_sign() && b.get_sign())
            }

            (Category::Infinity, Category::Infinity) => {
                if a.get_sign() ^ b.get_sign() ^ subtract {
                    return Self::nan(sem, a.get_sign() ^ b.get_sign());
                }
                Self::inf(sem, a.get_sign())
            }

            (Category::Normal, Category::Normal) => {
                // The IEEE 754 spec (section 6.3) states that cancellation
                // results in a positive zero, except for the case of the
                // negative rounding mode.
                let cancellation = subtract == (a.get_sign() == b.get_sign());
                let same_absolute_number = a.same_absolute_value(b);
                if cancellation && same_absolute_number {
                    let is_negative = RoundingMode::Negative == rm;
                    return Self::zero(sem, is_negative);
                }

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
    let a = Float::from_u64(FP64, 1);
    let b = Float::from_u64(FP64, 2);
    let _ = Float::add(a, b);
}

#[test]
fn test_addition() {
    fn add_helper(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        let c = Float::add(a, b);
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

    // Check that adding a negative and positive results in a positive zero for
    // the default rounding mode.
    let a = Float::from_f64(4.0);
    let b = Float::from_f64(-4.0);
    let c = Float::add(a.clone(), b);
    let d = Float::sub(a.clone(), a);
    assert!(c.is_zero());
    assert!(!c.is_negative());
    assert!(d.is_zero());
    assert!(!d.is_negative());
}

// Pg 120.  Chapter 4. Basic Properties and Algorithms.
#[test]
fn test_addition_large_numbers() {
    use super::float::FP64;
    let rm = RoundingMode::NearestTiesToEven;

    let one = Float::from_i64(FP64, 1);
    let mut a = Float::from_i64(FP64, 1);

    while Float::sub_with_rm(&Float::add_with_rm(&a, &one, rm), &a, rm) == one {
        a = Float::add_with_rm(&a, &a, rm);
    }

    let mut b = one.clone();
    while Float::sub_with_rm(&Float::add_with_rm(&a, &b, rm), &a, rm) != b {
        b = Float::add_with_rm(&b, &one, rm);
    }

    assert_eq!(a.as_f64(), 9007199254740992.);
    assert_eq!(b.as_f64(), 2.);
}

#[test]
fn add_denormals() {
    let v0 = f64::from_bits(0x0000_0000_0010_0010);
    let v1 = f64::from_bits(0x0000_0000_1001_0010);
    let v2 = f64::from_bits(0x1000_0000_0001_0010);
    assert_eq!(add_f64(v2, -v1), v2 - v1);

    let a0 = Float::from_f64(v0);
    assert_eq!(a0.as_f64(), v0);

    fn add_f64(a: f64, b: f64) -> f64 {
        let a0 = Float::from_f64(a);
        let b0 = Float::from_f64(b);
        assert_eq!(a0.as_f64(), a);
        Float::add(a0, b0).as_f64()
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

#[cfg(feature = "std")]
#[test]
fn add_special_values() {
    use crate::utils;

    // Test the addition of various irregular values.
    let values = utils::get_special_test_values();

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        Float::add(a, b).as_f64()
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

    let mut lfsr = utils::Lfsr::new();

    let v0: u64 = 0x645e91f69778bad3;
    let v1: u64 = 0xe4d91b16be9ae0c5;

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        let k = Float::add(a, b);
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

impl Float {
    /// Compute a*b using the rounding mode `rm`.
    pub fn mul_with_rm(a: &Self, b: &Self, rm: RoundingMode) -> Self {
        let sem = a.get_semantics();
        let sign = a.get_sign() ^ b.get_sign();

        // Table 8.4: Specification of multiplication for floating-point data of
        // positive sign. Page 251.
        match (a.get_category(), b.get_category()) {
            (Category::Zero, Category::NaN)
            | (Category::Normal, Category::NaN)
            | (Category::Infinity, Category::NaN) => {
                Self::nan(sem, b.get_sign())
            }
            (Category::NaN, Category::Infinity)
            | (Category::NaN, Category::NaN)
            | (Category::NaN, Category::Normal)
            | (Category::NaN, Category::Zero) => Self::nan(sem, a.get_sign()),
            (Category::Normal, Category::Infinity)
            | (Category::Infinity, Category::Normal)
            | (Category::Infinity, Category::Infinity) => Self::inf(sem, sign),
            (Category::Normal, Category::Zero)
            | (Category::Zero, Category::Normal)
            | (Category::Zero, Category::Zero) => Self::zero(sem, sign),

            (Category::Zero, Category::Infinity)
            | (Category::Infinity, Category::Zero) => Self::nan(sem, sign),

            (Category::Normal, Category::Normal) => {
                let (mut res, loss) = Self::mul_normals(a, b, sign);
                res.normalize(rm, loss);
                res
            }
        }
    }

    /// See Pg 251. 8.4 Floating-Point Multiplication
    fn mul_normals(a: &Self, b: &Self, sign: bool) -> (Self, LossFraction) {
        debug_assert_eq!(a.get_semantics(), b.get_semantics());
        let sem = a.get_semantics();
        // We multiply digits in the format 1.xx * 2^(e), or mantissa * 2^(e+1).
        // When we multiply two 2^(e+1) numbers, we get:
        // log(2^(e_a+1)*2^(e_b+1)) = e_a + e_b + 2.
        let mut exp = a.get_exp() + b.get_exp();

        let a_significand = a.get_mantissa();
        let b_significand = b.get_mantissa();
        let ab_significand = a_significand * b_significand;

        // The exponent is correct, but the bits are not in the right place.
        // Set the right exponent for where the bits are placed, and fix the
        // exponent below.
        exp -= sem.get_mantissa_len() as i64;

        let loss = LossFraction::ExactlyZero;
        (Self::from_parts(sem, sign, exp, ab_significand), loss)
    }
}

#[test]
fn test_mul_simple() {
    let a: f64 = -24.0;
    let b: f64 = 0.1;

    let af = Float::from_f64(a);
    let bf = Float::from_f64(b);
    let cf = Float::mul(af, bf);

    let r0 = cf.as_f64();
    let r1: f64 = a * b;
    assert_eq!(r0, r1);
}

#[test]
fn mul_regular_values() {
    // Test the addition of regular values.
    let values = [-5.0, 0., -0., 24., 1., 11., 10000., 256., 0.1, 3., 17.5];

    fn mul_f64(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        Float::mul(a, b).as_f64()
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

#[cfg(feature = "std")]
#[test]
fn test_mul_special_values() {
    use super::utils;

    // Test the multiplication of various irregular values.
    let values = utils::get_special_test_values();

    fn mul_f64(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        Float::mul(a, b).as_f64()
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

    let mut lfsr = utils::Lfsr::new();

    fn mul_f64(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        let k = Float::mul(a, b);
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

impl Float {
    /// Compute a/b, with the rounding mode `rm`.
    pub fn div_with_rm(a: &Self, b: &Self, rm: RoundingMode) -> Self {
        let sem = a.get_semantics();
        let sign = a.get_sign() ^ b.get_sign();
        // Table 8.5: Special values for x/y - Page 263.
        match (a.get_category(), b.get_category()) {
            (Category::NaN, _)
            | (_, Category::NaN)
            | (Category::Zero, Category::Zero)
            | (Category::Infinity, Category::Infinity) => Self::nan(sem, sign),

            (_, Category::Infinity) => Self::zero(sem, sign),
            (Category::Zero, _) => Self::zero(sem, sign),
            (_, Category::Zero) => Self::inf(sem, sign),
            (Category::Infinity, _) => Self::inf(sem, sign),
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
    fn div_normals(a: &Self, b: &Self) -> (Self, LossFraction) {
        debug_assert_eq!(a.get_semantics(), b.get_semantics());
        let sem = a.get_semantics();

        let mut a = a.clone();
        let mut b = b.clone();
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
        a_mantissa.shift_left(sem.get_mantissa_len());
        let reminder = a_mantissa.inplace_div(&b_mantissa);

        // Find 2 x reminder, to be able to compare to the reminder and figure
        // out the kind of loss that we have.
        let mut reminder_2x = reminder;
        reminder_2x.shift_left(1);

        let reminder = reminder_2x.cmp(&b_mantissa);
        let is_zero = reminder_2x.is_zero();
        let loss = match reminder {
            Ordering::Less => {
                if is_zero {
                    LossFraction::ExactlyZero
                } else {
                    LossFraction::LessThanHalf
                }
            }
            Ordering::Equal => LossFraction::ExactlyHalf,
            Ordering::Greater => LossFraction::MoreThanHalf,
        };

        let x = Self::from_parts(sem, sign, exp, a_mantissa);
        (x, loss)
    }
}

#[test]
fn test_div_simple() {
    let a: f64 = 1.0;
    let b: f64 = 7.0;

    let af = Float::from_f64(a);
    let bf = Float::from_f64(b);
    let cf = Float::div_with_rm(&af, &bf, RoundingMode::NearestTiesToEven);

    let r0 = cf.as_f64();
    let r1: f64 = a / b;
    assert_eq!(r0, r1);
}

#[cfg(feature = "std")]
#[test]
fn test_div_special_values() {
    use super::utils;

    // Test the multiplication of various irregular values.
    let values = utils::get_special_test_values();

    fn div_f64(a: f64, b: f64) -> f64 {
        let a = Float::from_f64(a);
        let b = Float::from_f64(b);
        Float::div_with_rm(&a, &b, RoundingMode::NearestTiesToEven).as_f64()
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

macro_rules! declare_operator {
    ($trait_name:ident,
     $func_name:ident,
     $func_impl_name:ident) => {
        // Self + Self
        impl $trait_name for Float {
            type Output = Self;
            fn $func_name(self, rhs: Self) -> Self {
                let sem = self.get_semantics();
                Self::$func_impl_name(&self, &rhs, sem.get_rounding_mode())
            }
        }

        // Self + u64
        impl $trait_name<u64> for Float {
            type Output = Self;
            fn $func_name(self, rhs: u64) -> Self {
                let sem = self.get_semantics();
                Self::$func_impl_name(
                    &self,
                    &Self::Output::from_u64(sem, rhs),
                    sem.get_rounding_mode(),
                )
            }
        }
        // &Self + &Self
        impl $trait_name<Self> for &Float {
            type Output = Float;
            fn $func_name(self, rhs: Self) -> Self::Output {
                let sem = self.get_semantics();
                Self::Output::$func_impl_name(
                    &self,
                    rhs,
                    sem.get_rounding_mode(),
                )
            }
        }
        // &Self + u64
        impl $trait_name<u64> for &Float {
            type Output = Float;
            fn $func_name(self, rhs: u64) -> Self::Output {
                let sem = self.get_semantics();
                Self::Output::$func_impl_name(
                    &self,
                    &Self::Output::from_u64(self.get_semantics(), rhs),
                    sem.get_rounding_mode(),
                )
            }
        }

        // &Self + Self
        impl $trait_name<Float> for &Float {
            type Output = Float;
            fn $func_name(self, rhs: Float) -> Self::Output {
                let sem = self.get_semantics();
                Self::Output::$func_impl_name(
                    &self,
                    &rhs,
                    sem.get_rounding_mode(),
                )
            }
        }
    };
}

declare_operator!(Add, add, add_with_rm);
declare_operator!(Sub, sub, sub_with_rm);
declare_operator!(Mul, mul, mul_with_rm);
declare_operator!(Div, div, div_with_rm);

macro_rules! declare_assign_operator {
    ($trait_name:ident,
     $func_name:ident,
     $func_impl_name:ident) => {
        impl $trait_name for Float {
            fn $func_name(&mut self, rhs: Self) {
                let sem = self.get_semantics();
                *self =
                    Self::$func_impl_name(self, &rhs, sem.get_rounding_mode());
            }
        }

        impl $trait_name<&Float> for Float {
            fn $func_name(&mut self, rhs: &Self) {
                let sem = self.get_semantics();
                *self =
                    Self::$func_impl_name(self, rhs, sem.get_rounding_mode());
            }
        }
    };
}

declare_assign_operator!(AddAssign, add_assign, add_with_rm);
declare_assign_operator!(SubAssign, sub_assign, sub_with_rm);
declare_assign_operator!(MulAssign, mul_assign, mul_with_rm);
declare_assign_operator!(DivAssign, div_assign, div_with_rm);

#[test]
fn test_operators() {
    use crate::FP64;
    let a = Float::from_f32(8.0).cast(FP64);
    let b = Float::from_f32(2.0).cast(FP64);
    let c = &a + &b;
    let d = &a - &b;
    let e = &a * &b;
    let f = &a / &b;
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
    let two = Float::from_f64(2.0).cast(FP128);
    let mut high = Float::from_f64(2.0).cast(FP128);
    let mut low = Float::from_f64(1.0).cast(FP128);

    for _ in 0..25 {
        let mid = (&high + &low) / 2;
        if (&mid * &mid) < two {
            low = mid;
        } else {
            high = mid;
        }
    }

    let res = low.cast(FP64);
    assert!(res.as_f64() < 1.4142137_f64);
    assert!(res.as_f64() > 1.4142134_f64);
}

#[cfg(feature = "std")]
#[test]
fn test_famous_pentium4_bug() {
    use crate::std::string::ToString;
    // https://en.wikipedia.org/wiki/Pentium_FDIV_bug
    use crate::FP128;

    let a = Float::from_u64(FP128, 4_195_835);
    let b = Float::from_u64(FP128, 3_145_727);
    let res = a / b;
    let result = res.to_string();
    assert!(result.starts_with("1.333820449136241002"));
}

impl Float {
    // Perform a fused multiply-add of normal numbers, without rounding.
    fn fused_mul_add_normals(
        a: &Self,
        b: &Self,
        c: &Self,
    ) -> (Self, LossFraction) {
        debug_assert_eq!(a.get_semantics(), b.get_semantics());
        let sem = a.get_semantics();

        // Multiply a and b, without rounding.
        let sign = a.get_sign() ^ b.get_sign();
        let mut ab = Self::mul_normals(a, b, sign).0;

        // Shift the product, to allow enough precision for the addition.
        // Notice that this can be implemented more efficiently with 3 extra
        // bits and sticky bits.
        // See 8.5. Floating-Point Fused Multiply-Add, Page 255.
        let mut c = c.clone();
        let extra_bits = sem.get_precision() + 1;
        ab.shift_significand_left(extra_bits as u64);
        c.shift_significand_left(extra_bits as u64);

        // Perform the addition, without rounding.
        Self::add_or_sub_normals(&ab, &c, false)
    }

    /// Compute a*b + c, with the rounding mode `rm`.
    pub fn fused_mul_add_with_rm(
        a: &Self,
        b: &Self,
        c: &Self,
        rm: RoundingMode,
    ) -> Self {
        if a.is_normal() && b.is_normal() && c.is_normal() {
            let (mut res, loss) = Self::fused_mul_add_normals(a, b, c);
            res.normalize(rm, loss); // Finally, round the result.
            res
        } else {
            // Perform two operations. First, handle non-normal values.

            // NaN anything = NaN
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return Self::nan(a.get_semantics(), a.get_sign());
            }
            // (infinity * 0) + c = NaN
            if (a.is_inf() && b.is_zero()) || (a.is_zero() && b.is_inf()) {
                return Self::nan(a.get_semantics(), a.get_sign());
            }
            // (normal * normal) + infinity = infinity
            if a.is_normal() && b.is_normal() && c.is_inf() {
                return c.clone();
            }
            // (normal * 0) + c = c
            if a.is_zero() || b.is_zero() {
                return c.clone();
            }

            // Multiply (with rounding), and add (with rounding).
            let ab = Self::mul_with_rm(a, b, rm);
            Self::add_with_rm(&ab, c, rm)
        }
    }

    /// Compute a*b + c.
    pub fn fma(a: &Self, b: &Self, c: &Self) -> Self {
        Self::fused_mul_add_with_rm(a, b, c, c.get_rounding_mode())
    }
}

#[test]
fn test_fma() {
    let v0 = -10.;
    let v1 = -1.1;
    let v2 = 0.000000000000000000000000000000000000001;
    let af = Float::from_f64(v0);
    let bf = Float::from_f64(v1);
    let cf = Float::from_f64(v2);

    let r = Float::fused_mul_add_with_rm(
        &af,
        &bf,
        &cf,
        RoundingMode::NearestTiesToEven,
    );

    assert_eq!(f64::mul_add(v0, v1, v2), r.as_f64());
}

#[cfg(feature = "std")]
#[test]
fn test_fma_simple() {
    use super::utils;
    // Test the multiplication of various irregular values.
    let values = utils::get_special_test_values();
    for a in values {
        for b in values {
            for c in values {
                let af = Float::from_f64(a);
                let bf = Float::from_f64(b);
                let cf = Float::from_f64(c);

                let rf = Float::fused_mul_add_with_rm(
                    &af,
                    &bf,
                    &cf,
                    RoundingMode::NearestTiesToEven,
                );

                let r0 = rf.as_f64();
                let r1: f64 = a.mul_add(b, c);
                assert_eq!(r0.is_finite(), r1.is_finite());
                assert_eq!(r0.is_nan(), r1.is_nan());
                assert_eq!(r0.is_infinite(), r1.is_infinite());
                // Check that the results are bit identical, or are both NaN.
                assert!(r1.is_nan() || r1.is_infinite() || r0 == r1);
            }
        }
    }
}

#[test]
fn test_fma_random_vals() {
    use super::utils;

    let mut lfsr = utils::Lfsr::new();

    fn mul_f32(a: f32, b: f32, c: f32) -> f32 {
        let a = Float::from_f32(a);
        let b = Float::from_f32(b);
        let c = Float::from_f32(c);
        let k = Float::fused_mul_add_with_rm(
            &a,
            &b,
            &c,
            RoundingMode::NearestTiesToEven,
        );
        k.as_f32()
    }

    for _ in 0..50000 {
        let v0 = lfsr.get64() as u32;
        let v1 = lfsr.get64() as u32;
        let v2 = lfsr.get64() as u32;

        let f0 = f32::from_bits(v0);
        let f1 = f32::from_bits(v1);
        let f2 = f32::from_bits(v2);

        let r0 = mul_f32(f0, f1, f2);
        let r1 = f32::mul_add(f0, f1, f2);
        assert_eq!(r0.is_finite(), r1.is_finite());
        assert_eq!(r0.is_nan(), r1.is_nan());
        assert_eq!(r0.is_infinite(), r1.is_infinite());
        let r0_bits = r0.to_bits();
        let r1_bits = r1.to_bits();
        // Check that the results are bit identical, or are both NaN.
        assert!(r1.is_nan() || r0_bits == r1_bits);
    }
}
