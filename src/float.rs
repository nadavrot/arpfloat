//! This module contains the Float data structure and basic methods.

extern crate alloc;
use super::bigint::BigInt;
use super::bigint::LossFraction;
use core::cmp::Ordering;

/// Defines the supported rounding modes.
/// See IEEE754-2019 Section 4.3 Rounding-direction attributes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundingMode {
    None,
    NearestTiesToEven,
    NearestTiesToAway,
    Zero,
    Positive,
    Negative,
}

impl RoundingMode {
    /// Create a rounding mode from a string, if valid, or return none.
    pub fn from_string(s: &str) -> Option<Self> {
        match s {
            "NearestTiesToEven" => Some(RoundingMode::NearestTiesToEven),
            "NearestTiesToAway" => Some(RoundingMode::NearestTiesToAway),
            "Zero" => Some(RoundingMode::Zero),
            "Positive" => Some(RoundingMode::Positive),
            "Negative" => Some(RoundingMode::Negative),
            _ => None,
        }
    }
}

/// Controls the semantics of a floating point number with:
/// 'precision', that determines the number of bits, 'exponent' that controls
/// the dynamic range of the number, and rounding mode that controls how
/// rounding is done after arithmetic operations.
///
/// # Example
///
/// ```
///     use arpfloat::{Float, RoundingMode, Semantics};
///
///     // Create a new floating point semantics.
///     let sem = Semantics::new(10, 100, RoundingMode::Positive);
///     // Create the number 1.0 with the new semantics.
///     let x = Float::one(sem, false);
///
///     // Check that the value is correct when casting to `double`.
///     assert_eq!(x.as_f64(), 1.0);
/// ```

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Semantics {
    /// The number of bits that define the range of the exponent.
    pub exponent: usize,
    /// The number of bits in the significand (mantissa + 1).
    pub precision: usize,
    /// The rounding mode used when performing operations on this type.
    pub mode: RoundingMode,
}

impl Semantics {
    pub const fn new(
        exponent: usize,
        precision: usize,
        mode: RoundingMode,
    ) -> Self {
        Semantics {
            exponent,
            precision,
            mode,
        }
    }
    /// Returns the precision in bits.
    pub fn get_precision(&self) -> usize {
        self.precision
    }
    /// Returns the length of the mantissa in bits (precision - 1).
    pub fn get_mantissa_len(&self) -> usize {
        self.precision - 1
    }
    /// Returns the length of the exponent in bits, which defines the valid
    /// range.
    pub fn get_exponent_len(&self) -> usize {
        self.exponent
    }

    /// Returns the rounding mode of the type.
    pub fn get_rounding_mode(&self) -> RoundingMode {
        self.mode
    }

    /// Create a new float semantics with increased precision with 'add'
    /// additional digits.
    pub fn increase_precision(&self, more: usize) -> Semantics {
        Semantics::new(self.exponent, self.precision + more, self.mode)
    }
    /// Create a new float semantics with increased precision with 'add'
    /// additional digits, plus ceil(log2) of the number.
    pub fn grow_log(&self, more: usize) -> Semantics {
        let log2 = self.log_precision();
        Semantics::new(self.exponent, self.precision + more + log2, self.mode)
    }

    /// Return a log2 approximation for the precision value.
    pub fn log_precision(&self) -> usize {
        // This is ~Log2(precision)
        64 - (self.precision as u64).leading_zeros() as usize
    }

    /// Create a new float semantics with increased exponent with 'more'
    /// additional digits.
    pub fn increase_exponent(&self, more: usize) -> Semantics {
        Semantics::new(self.exponent + more, self.precision, self.mode)
    }
    /// Create a new float semantics with a different rounding mode 'mode'.
    pub fn with_rm(&self, rm: RoundingMode) -> Semantics {
        Semantics::new(self.exponent, self.precision, rm)
    }

    /// Returns the exponent bias for the number, as a positive number.
    /// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
    pub(crate) fn get_bias(&self) -> i64 {
        let e = self.get_exponent_len();
        ((1u64 << (e - 1)) - 1) as i64
    }
    /// Returns the upper and lower bounds of the exponent.
    pub fn get_exp_bounds(&self) -> (i64, i64) {
        let exp_min: i64 = -self.get_bias() + 1;
        // The highest value is 0xFFFE, because 0xFFFF is used for signaling.
        let exp_max: i64 = (1 << self.get_exponent_len()) - self.get_bias() - 2;
        (exp_min, exp_max)
    }
}

/// Declare the different categories of the floating point number. These
/// categories are internal to the float, and can be access by the accessors:
/// is_inf, is_zero, is_nan, is_normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Category {
    Infinity,
    NaN,
    Normal,
    Zero,
}

/// This is the main data structure of this library. It represents an
/// arbitrary-precision floating-point number.
#[derive(Debug, Clone)]
pub struct Float {
    // The semantics of the float (precision, exponent range).
    sem: Semantics,
    // The Sign bit.
    sign: bool,
    // The Exponent.
    exp: i64,
    // The significand, including the implicit bit, aligned to the right.
    // Format [00000001xxxxxxx].
    mantissa: BigInt,
    // The kind of number this float represents.
    category: Category,
}

impl Float {
    pub(crate) fn get_mantissa_len(&self) -> usize {
        self.sem.get_mantissa_len()
    }
    pub(crate) fn get_exponent_len(&self) -> usize {
        self.sem.get_exponent_len()
    }

    /// Create a new normal floating point number.
    pub fn from_parts(
        sem: Semantics,
        sign: bool,
        exp: i64,
        mantissa: BigInt,
    ) -> Self {
        if mantissa.is_zero() {
            return Float::zero(sem, sign);
        }
        Float {
            sem,
            sign,
            exp,
            mantissa,
            category: Category::Normal,
        }
    }

    /// Create a new normal floating point number.
    pub(crate) fn raw(
        sem: Semantics,
        sign: bool,
        exp: i64,
        mantissa: BigInt,
        category: Category,
    ) -> Self {
        Float {
            sem,
            sign,
            exp,
            mantissa,
            category,
        }
    }

    /// Returns a new zero float.
    pub fn zero(sem: Semantics, sign: bool) -> Self {
        Float {
            sem,
            sign,
            exp: 0,
            mantissa: BigInt::zero(),
            category: Category::Zero,
        }
    }

    /// Returns a new float with the value one.
    pub fn one(sem: Semantics, sign: bool) -> Self {
        let mut one = BigInt::one();
        one.shift_left(sem.get_mantissa_len());
        Float {
            sem,
            sign,
            exp: 0,
            mantissa: one,
            category: Category::Normal,
        }
    }

    /// Returns a new infinity float.
    pub fn inf(sem: Semantics, sign: bool) -> Self {
        Float {
            sem,
            sign,
            exp: 0,
            mantissa: BigInt::zero(),
            category: Category::Infinity,
        }
    }

    /// Returns a new NaN float.
    pub fn nan(sem: Semantics, sign: bool) -> Self {
        Float {
            sem,
            sign,
            exp: 0,
            mantissa: BigInt::zero(),
            category: Category::NaN,
        }
    }
    /// Returns true if the Float is negative
    pub fn is_negative(&self) -> bool {
        self.sign
    }

    /// Returns true if the Float is +-inf.
    pub fn is_inf(&self) -> bool {
        if let Category::Infinity = self.category {
            return true;
        }
        false
    }

    /// Returns true if the Float is a +- NaN.
    pub fn is_nan(&self) -> bool {
        if let Category::NaN = self.category {
            return true;
        }
        false
    }

    /// Returns true if the Float is a +- zero.
    pub fn is_zero(&self) -> bool {
        if let Category::Zero = self.category {
            return true;
        }
        false
    }

    /// Returns true if this number is normal (not Zero, Nan, Inf).
    pub fn is_normal(&self) -> bool {
        if let Category::Normal = self.category {
            return true;
        }
        false
    }

    /// Return the semantics of the number
    pub fn get_semantics(&self) -> Semantics {
        self.sem
    }

    /// Returns the rounding mode of the number.
    pub fn get_rounding_mode(&self) -> RoundingMode {
        self.sem.get_rounding_mode()
    }

    /// Update the sign of the float to `sign`. True means negative.
    pub fn set_sign(&mut self, sign: bool) {
        self.sign = sign
    }

    /// Returns the sign of the float. True means negative.
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    /// Returns the mantissa of the float.
    pub fn get_mantissa(&self) -> BigInt {
        self.mantissa.clone()
    }

    /// Returns the exponent of the float.
    pub fn get_exp(&self) -> i64 {
        self.exp
    }

    /// Returns the category of the float.
    pub fn get_category(&self) -> Category {
        self.category
    }

    /// Returns a new float which has a flipped sign (negated value).
    pub fn neg(&self) -> Self {
        Self::raw(
            self.sem,
            !self.sign,
            self.exp,
            self.mantissa.clone(),
            self.category,
        )
    }

    /// Shift the mantissa to the left to ensure that the MSB if the mantissa
    /// is set to the precision. The method updates the exponent to keep the
    /// number correct.
    pub(super) fn align_mantissa(&mut self) {
        let bits =
            self.sem.get_precision() as i64 - self.mantissa.msb_index() as i64;
        if bits > 0 {
            self.exp += bits;
            self.mantissa.shift_left(bits as usize);
        }
    }

    /// Prints the number using the internal representation.
    #[cfg(feature = "std")]
    pub fn dump(&self) {
        use std::println;
        let sign = if self.sign { "-" } else { "+" };
        match self.category {
            Category::NaN => {
                println!("[{}NaN]", sign);
            }
            Category::Infinity => {
                println!("[{}Inf]", sign);
            }
            Category::Zero => {
                println!("[{}0.0]", sign);
            }
            Category::Normal => {
                let m = self.mantissa.as_binary();
                println!("FP[{} E={:4} M = {}]", sign, self.exp, m);
            }
        }
    }

    #[cfg(not(feature = "std"))]
    pub fn dump(&self) {
        // No-op in no_std environments
    }

    /// Returns the exponent bias for the number, as a positive number.
    /// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
    pub(crate) fn get_bias(&self) -> i64 {
        self.sem.get_bias()
    }

    /// Returns the upper and lower bounds of the exponent.
    pub fn get_exp_bounds(&self) -> (i64, i64) {
        self.sem.get_exp_bounds()
    }
}

// IEEE 754-2019
// Table 3.5 — Binary interchange format parameters.
use RoundingMode::NearestTiesToEven as nte;

/// Predefined BF16 float with 8 exponent bits, and 7 mantissa bits.
pub const BF16: Semantics = Semantics::new(8, 8, nte);
/// Predefined FP16 float with 5 exponent bits, and 10 mantissa bits.
pub const FP16: Semantics = Semantics::new(5, 11, nte);
/// Predefined FP32 float with 8 exponent bits, and 23 mantissa bits.
pub const FP32: Semantics = Semantics::new(8, 24, nte);
/// Predefined FP64 float with 11 exponent bits, and 52 mantissa bits.
pub const FP64: Semantics = Semantics::new(11, 53, nte);
/// Predefined FP128 float with 15 exponent bits, and 112 mantissa bits.
pub const FP128: Semantics = Semantics::new(15, 113, nte);
/// Predefined FP256 float with 19 exponent bits, and 236 mantissa bits.
pub const FP256: Semantics = Semantics::new(19, 237, nte);

/// Shift `val` by `bits`, and report the loss.
pub(crate) fn shift_right_with_loss(
    val: &BigInt,
    bits: usize,
) -> (BigInt, LossFraction) {
    let mut val = val.clone();
    let loss = val.get_loss_kind_for_bit(bits);
    val.shift_right(bits);
    (val, loss)
}

/// Combine the loss of accuracy with `msb` more significant and `lsb`
/// less significant.
fn combine_loss_fraction(msb: LossFraction, lsb: LossFraction) -> LossFraction {
    if !lsb.is_exactly_zero() {
        if msb.is_exactly_zero() {
            return LossFraction::LessThanHalf;
        } else if msb.is_exactly_half() {
            return LossFraction::MoreThanHalf;
        }
    }
    msb
}

#[test]
fn shift_right_fraction() {
    let x: BigInt = BigInt::from_u64(0b10000000);
    let res = shift_right_with_loss(&x, 3);
    assert!(res.1.is_exactly_zero());

    let x: BigInt = BigInt::from_u64(0b10000111);
    let res = shift_right_with_loss(&x, 3);
    assert!(res.1.is_mt_half());

    let x: BigInt = BigInt::from_u64(0b10000100);
    let res = shift_right_with_loss(&x, 3);
    assert!(res.1.is_exactly_half());

    let x: BigInt = BigInt::from_u64(0b10000001);
    let res = shift_right_with_loss(&x, 3);
    assert!(res.1.is_lt_half());
}

impl Float {
    /// The number overflowed, set the right value based on the rounding mode
    /// and sign.
    fn overflow(&mut self, rm: RoundingMode) {
        let bounds = self.get_exp_bounds();
        let inf = Self::inf(self.sem, self.sign);
        let max = Self::from_parts(
            self.sem,
            self.sign,
            bounds.1,
            BigInt::all1s(self.get_mantissa_len()),
        );

        *self = match rm {
            RoundingMode::None => inf,
            RoundingMode::NearestTiesToEven => inf,
            RoundingMode::NearestTiesToAway => inf,
            RoundingMode::Zero => max,
            RoundingMode::Positive => {
                if self.sign {
                    max
                } else {
                    inf
                }
            }
            RoundingMode::Negative => {
                if self.sign {
                    inf
                } else {
                    max
                }
            }
        }
    }

    /// Verify that the exponent is legal.
    pub(crate) fn check_bounds(&self) {
        let bounds = self.get_exp_bounds();
        debug_assert!(self.exp >= bounds.0);
        debug_assert!(self.exp <= bounds.1);
        let max_mantissa = BigInt::one_hot(self.sem.get_precision());
        debug_assert!(self.mantissa.lt(&max_mantissa));
    }

    pub(crate) fn shift_significand_left(&mut self, amt: u64) {
        self.exp -= amt as i64;
        self.mantissa.shift_left(amt as usize);
    }

    pub(crate) fn shift_significand_right(&mut self, amt: u64) -> LossFraction {
        self.exp += amt as i64;
        let res = shift_right_with_loss(&self.mantissa, amt as usize);
        self.mantissa = res.0;
        res.1
    }

    /// Returns true if we need to round away from zero (increment the mantissa).
    pub(crate) fn need_round_away_from_zero(
        &self,
        rm: RoundingMode,
        loss: LossFraction,
    ) -> bool {
        debug_assert!(self.is_normal() || self.is_zero());
        match rm {
            RoundingMode::Positive => !self.sign,
            RoundingMode::Negative => self.sign,
            RoundingMode::Zero => false,
            RoundingMode::None => false,
            RoundingMode::NearestTiesToAway => loss.is_gte_half(),
            RoundingMode::NearestTiesToEven => {
                if loss.is_mt_half() {
                    return true;
                }

                loss.is_exactly_half() && self.mantissa.is_odd()
            }
        }
    }

    /// Returns true if the absolute value of the two numbers are the same.
    pub(crate) fn same_absolute_value(&self, other: &Self) -> bool {
        if self.category != other.category {
            return false;
        }
        match self.category {
            Category::Infinity => true,
            Category::NaN => true,
            Category::Zero => true,
            Category::Normal => {
                self.exp == other.exp && self.mantissa == other.mantissa
            }
        }
    }

    /// Normalize the number by adjusting the exponent to the legal range, shift
    /// the mantissa to the msb, and round the number if bits are lost. This is
    /// based on Neil Booth' implementation in APFloat.
    pub(crate) fn normalize(&mut self, rm: RoundingMode, loss: LossFraction) {
        if !self.is_normal() {
            return;
        }
        let mut loss = loss;
        let bounds = self.get_exp_bounds();

        let nmsb = self.mantissa.msb_index() as i64;

        // Step I - adjust the exponent.
        if nmsb > 0 {
            // Align the number so that the MSB bit will be MANTISSA + 1.
            let mut exp_change = nmsb - self.sem.get_precision() as i64;

            // Handle overflowing exponents.
            if self.exp + exp_change > bounds.1 {
                self.overflow(rm);
                self.check_bounds();
                return;
            }

            // Handle underflowing low exponents. Don't allow to go below the
            // legal exponent range.
            if self.exp + exp_change < bounds.0 {
                exp_change = bounds.0 - self.exp;
            }

            if exp_change < 0 {
                // Handle reducing the exponent.
                debug_assert!(loss.is_exactly_zero(), "losing information");
                self.shift_significand_left(-exp_change as u64);
                return;
            }

            if exp_change > 0 {
                // Handle increasing the exponent.
                let loss2 = self.shift_significand_right(exp_change as u64);
                loss = combine_loss_fraction(loss2, loss);
            }
        }

        //Step II - round the number.

        // If nothing moved or the shift didn't mess things up then we're done.
        if loss.is_exactly_zero() {
            // Canonicalize to zero.
            if self.mantissa.is_zero() {
                *self = Self::zero(self.sem, self.sign);
                return;
            }
            return;
        }

        // Check if we need to round away from zero.
        if self.need_round_away_from_zero(rm, loss) {
            if self.mantissa.is_zero() {
                self.exp = bounds.0
            }

            let one = BigInt::one();
            self.mantissa = self.mantissa.clone() + one;
            // Did the mantissa overflow?
            let mut m = self.mantissa.clone();
            m.shift_right(self.sem.get_precision());
            if !m.is_zero() {
                // Can we fix the exponent?
                if self.exp < bounds.1 {
                    self.shift_significand_right(1);
                } else {
                    *self = Self::inf(self.sem, self.sign);
                    return;
                }
            }
        }

        // Canonicalize.
        if self.mantissa.is_zero() {
            *self = Self::zero(self.sem, self.sign);
        }
    } // round.
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        let bitwise = self.sign == other.sign
            && self.exp == other.exp
            && self.mantissa == other.mantissa
            && self.category == other.category;

        match self.category {
            Category::Infinity | Category::Normal => bitwise,
            Category::Zero => other.is_zero(),
            Category::NaN => false,
        }
    }
}

/// Page 66. Chapter 3. Floating-Point Formats and Environment
/// Table 3.8: Comparison predicates and the four relations.
///   and
/// IEEE 754-2019 section 5.10 - totalOrder.
impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        debug_assert_eq!(self.get_semantics(), other.get_semantics());
        let bool_to_ord = |ord: bool| -> Option<Ordering> {
            if ord {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            }
        };

        match (self.category, other.category) {
            (Category::NaN, _) | (_, Category::NaN) => None,
            (Category::Zero, Category::Zero) => Some(Ordering::Equal),
            (Category::Infinity, Category::Infinity) => {
                if self.sign == other.sign {
                    Some(Ordering::Equal)
                } else {
                    bool_to_ord(self.sign)
                }
            }
            (Category::Infinity, Category::Normal)
            | (Category::Infinity, Category::Zero)
            | (Category::Normal, Category::Zero) => bool_to_ord(self.sign),

            (Category::Normal, Category::Infinity)
            | (Category::Zero, Category::Infinity)
            | (Category::Zero, Category::Normal) => bool_to_ord(!other.sign),

            (Category::Normal, Category::Normal) => {
                if self.sign != other.sign {
                    bool_to_ord(self.sign)
                } else if self.exp < other.exp {
                    bool_to_ord(!self.sign)
                } else if self.exp > other.exp {
                    bool_to_ord(self.sign)
                } else {
                    match self.mantissa.cmp(&other.mantissa) {
                        Ordering::Less => bool_to_ord(!self.sign),
                        Ordering::Equal => Some(Ordering::Equal),
                        Ordering::Greater => bool_to_ord(self.sign),
                    }
                }
            }
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn test_comparisons() {
    use super::utils;

    // Compare a bunch of special values, using the <,>,== operators and check
    // that they match the comparison on doubles.
    for first in utils::get_special_test_values() {
        for second in utils::get_special_test_values() {
            let is_less = first < second;
            let is_eq = first == second;
            let is_gt = first > second;
            let first = Float::from_f64(first);
            let second = Float::from_f64(second);
            assert_eq!(is_less, first < second, "<");
            assert_eq!(is_eq, first == second, "==");
            assert_eq!(is_gt, first > second, ">");
        }
    }
}

#[test]
fn test_one_imm() {
    let sem = Semantics::new(10, 12, nte);
    let x = Float::one(sem, false);
    assert_eq!(x.as_f64(), 1.0);
}

#[test]
pub fn test_bigint_ctor() {
    // Make sure that we can load numbers of the highest border of the FP16
    // number.
    let bi = BigInt::from_u64(65519);
    assert_eq!(Float::from_bigint(FP16, bi).cast(FP32).to_i64(), 65504);
    assert_eq!(Float::from_f64(65519.).cast(FP16).to_i64(), 65504);

    // Make sure that we can load numbers that are greater than the precision
    // and that normalization fixes and moves things to the right place.
    let sem = Semantics::new(40, 10, nte);
    let bi = BigInt::from_u64(1 << 14);
    let num = Float::from_bigint(sem, bi);
    assert_eq!(num.to_i64(), 1 << 14);
}

#[test]
pub fn test_semantics_size() {
    assert_eq!(FP16.log_precision(), 4);
    assert_eq!(FP32.log_precision(), 5);
    assert_eq!(FP64.log_precision(), 6);
    assert_eq!(FP128.log_precision(), 7);
}

impl Semantics {
    /// Returns the maximum value of the number.
    pub fn get_max_positive_value(&self) -> Float {
        let exp = self.get_exp_bounds().1;
        let mantissa = BigInt::all1s(self.get_precision());
        Float::from_parts(*self, false, exp, mantissa)
    }

    /// Returns the minimum positive value of the number (subnormal).
    /// See https://en.wikipedia.org/wiki/IEEE_754
    pub fn get_min_positive_value(&self) -> Float {
        let exp = self.get_exp_bounds().0;
        let mantissa = BigInt::one();
        Float::from_parts(*self, false, exp, mantissa)
    }

    /// Returns true if the number can be represented exactly in this format.
    /// A number can be represented exactly if the exponent is in the range, and
    /// the mantissa is not too large. In other words, the number 'val' can be
    /// converted to this format without any loss of accuracy.
    pub fn can_represent_exactly(&self, val: &Float) -> bool {
        // Can always represent Inf, NaN, Zero.
        if !val.is_normal() {
            return true;
        }

        // Check the semantics of the other value.
        let other_sem = val.get_semantics();
        if other_sem.get_precision() <= self.get_precision()
            && other_sem.get_exponent_len() <= self.get_exponent_len()
        {
            return true;
        }

        // Check the exponent value.
        let exp = val.get_exp();
        let bounds = self.get_exp_bounds();
        if exp < bounds.0 || exp > bounds.1 {
            return false;
        }

        // Check if the mantissa is zero.
        if val.get_mantissa().is_zero() {
            return true;
        }

        // Check how much we can shift-right the number without losing bits.
        let last = val.get_mantissa().trailing_zeros();
        let first = val.get_mantissa().msb_index();
        // Notice that msb_index is 1-based, but this is okay because we want to
        // count the number of bits including the last.
        let used_bits = first - last;
        used_bits <= self.get_precision()
    }
}

#[test]
fn test_min_max_val() {
    assert_eq!(FP16.get_max_positive_value().as_f64(), 65504.0);
    assert_eq!(FP32.get_max_positive_value().as_f64(), f32::MAX as f64);
    assert_eq!(FP64.get_max_positive_value().as_f64(), f64::MAX);
    assert_eq!(FP32.get_min_positive_value().as_f32(), f32::from_bits(0b01));
    assert_eq!(FP64.get_min_positive_value().as_f64(), f64::from_bits(0b01));
}

#[test]
fn test_can_represent_exactly() {
    assert!(FP16.can_represent_exactly(&Float::from_f64(1.0)));
    assert!(FP16.can_represent_exactly(&Float::from_f64(65504.0)));
    assert!(!FP16.can_represent_exactly(&Float::from_f64(65504.1)));
    assert!(!FP16.can_represent_exactly(&Float::from_f64(0.0001)));

    let m10 = BigInt::from_u64(0b1000000001);
    let m11 = BigInt::from_u64(0b10000000001);
    let m12 = BigInt::from_u64(0b100000000001);

    let val10bits = Float::from_parts(FP32, false, 0, m10);
    let val11bits = Float::from_parts(FP32, false, 0, m11);
    let val12bits = Float::from_parts(FP32, false, 0, m12);

    assert!(FP16.can_represent_exactly(&val10bits));
    assert!(FP16.can_represent_exactly(&val11bits));
    assert!(!FP16.can_represent_exactly(&val12bits));

    assert!(FP32.can_represent_exactly(&Float::pi(FP32)));
    assert!(!FP32.can_represent_exactly(&Float::pi(FP64)));
    assert!(FP64.can_represent_exactly(&Float::pi(FP32)));
}
