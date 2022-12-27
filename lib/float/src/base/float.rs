use super::bigint::BigInt;
use super::bigint::LossFraction;

#[derive(Debug, Clone, Copy)]
pub enum RoundingMode {
    NearestTiesToEven,
    NearestTiesToAway,
    Zero,
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Category {
    Infinity,
    NaN,
    Normal,
    Zero,
}

pub type MantissaTy = BigInt<6>;

#[derive(Debug, Clone, Copy)]
pub struct Float<const EXPONENT: usize, const MANTISSA: usize> {
    // The Sign bit.
    sign: bool,
    // The Exponent.
    exp: i64,
    // The significand, including the implicit bit, aligned to the right.
    // Format [00000001xxxxxxx].
    mantissa: MantissaTy,
    // The kind of number this float represents.
    category: Category,
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Create a new normal floating point number.
    pub fn new(sign: bool, exp: i64, mantissa: MantissaTy) -> Self {
        if mantissa.is_zero() {
            return Float::zero(sign);
        }
        Float {
            sign,
            exp,
            mantissa,
            category: Category::Normal,
        }
    }

    /// Create a new normal floating point number.
    pub fn raw(
        sign: bool,
        exp: i64,
        mantissa: MantissaTy,
        category: Category,
    ) -> Self {
        Float {
            sign,
            exp,
            mantissa,
            category,
        }
    }

    /// \returns a new zero float.
    pub fn zero(sign: bool) -> Self {
        Float {
            sign,
            exp: 0,
            mantissa: MantissaTy::zero(),
            category: Category::Zero,
        }
    }

    /// \returns a new infinity float.
    pub fn inf(sign: bool) -> Self {
        Float {
            sign,
            exp: 0,
            mantissa: MantissaTy::zero(),
            category: Category::Infinity,
        }
    }

    /// \returns a new NaN float.
    pub fn nan(sign: bool) -> Self {
        Float {
            sign,
            exp: 0,
            mantissa: MantissaTy::zero(),
            category: Category::NaN,
        }
    }
    /// \returns True if the Float is negative
    pub fn is_negative(&self) -> bool {
        self.sign
    }

    /// \returns True if the Float is +-inf.
    pub fn is_inf(&self) -> bool {
        if let Category::Infinity = self.category {
            return true;
        }
        false
    }

    /// \returns True if the Float is a +- NaN.
    pub fn is_nan(&self) -> bool {
        if let Category::NaN = self.category {
            return true;
        }
        false
    }

    /// \returns True if the Float is a +- NaN.
    pub fn is_zero(&self) -> bool {
        if let Category::Zero = self.category {
            return true;
        }
        false
    }

    pub fn is_normal(&self) -> bool {
        if let Category::Normal = self.category {
            return true;
        }
        false
    }

    pub fn set_sign(&mut self, sign: bool) {
        self.sign = sign
    }

    pub fn get_sign(&self) -> bool {
        self.sign
    }

    pub fn get_mantissa(&self) -> MantissaTy {
        self.mantissa
    }

    pub fn get_exp(&self) -> i64 {
        self.exp
    }

    pub fn get_category(&self) -> Category {
        self.category
    }

    pub fn neg(&self) -> Self {
        Self::raw(!self.sign, self.exp, self.mantissa, self.category)
    }

    /// \returns True if abs(self) < abs(other).
    pub fn absolute_less_than(&self, other: Self) -> bool {
        assert!(self.is_normal());
        let mc = self.mantissa.cmp(&other.get_mantissa());
        match self.exp.cmp(&other.get_exp()) {
            Ordering::Less => true,
            Ordering::Equal => mc.is_lt(),
            Ordering::Greater => false,
        }
    }

    pub fn dump(&self) {
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
                let m = self.mantissa;
                println!(
                    "FP[{} E={:4} M = 0x{:64b}]",
                    sign,
                    self.exp,
                    m.to_u128()
                );
            }
        }
    }

    // \returns the bias for this Float type.
    pub fn compute_ieee745_bias(exponent_bits: usize) -> usize {
        (1 << (exponent_bits - 1)) - 1
    }

    /// Returns the exponent bias for the number, as a positive number.
    /// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
    pub fn get_bias() -> i64 {
        Self::compute_ieee745_bias(EXPONENT) as i64
    }

    /// \returns the upper and lower bounds of the exponent.
    pub fn get_exp_bounds() -> (i64, i64) {
        let exp_min: i64 = -Self::get_bias() + 1;
        // The highest value is 0xFFFE, because 0xFFFF is used for signaling.
        let exp_max: i64 = (1 << EXPONENT) - Self::get_bias() - 2;
        (exp_min, exp_max)
    }

    /// \returns the number of bits in the significand, including the integer
    /// part.
    pub fn get_precision() -> u64 {
        (MANTISSA + 1) as u64
    }
}

pub type FP16 = Float<5, 10>;
pub type FP32 = Float<8, 23>;
pub type FP64 = Float<11, 52>;
pub type FP128 = Float<15, 112>;

//// Shift \p val by \p bits, and report the loss.
pub fn shift_right_with_loss(
    mut val: MantissaTy,
    bits: u64,
) -> (MantissaTy, LossFraction) {
    let loss = val.get_loss_kind_for_bit(bits as usize);
    val.shift_right(bits as usize);
    (val, loss)
}

/// Combine the loss of accuracy with \p msb more significant and \p lsb
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
    let x = MantissaTy::from_u64(0b10000000);
    let res = shift_right_with_loss(x, 3);
    assert!(res.1.is_exactly_zero());

    let x = MantissaTy::from_u64(0b10000111);
    let res = shift_right_with_loss(x, 3);
    assert!(res.1.is_mt_half());

    let x = MantissaTy::from_u64(0b10000100);
    let res = shift_right_with_loss(x, 3);
    assert!(res.1.is_exactly_half());

    let x = MantissaTy::from_u64(0b10000001);
    let res = shift_right_with_loss(x, 3);
    assert!(res.1.is_lt_half());
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    fn overflow(&mut self, rm: RoundingMode) {
        let bounds = Self::get_exp_bounds();
        let inf = Self::inf(self.sign);
        let max = Self::new(self.sign, bounds.1, MantissaTy::all1s(MANTISSA));

        *self = match rm {
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

    pub fn check_bounds(&self) {
        let bounds = Self::get_exp_bounds();
        assert!(self.exp >= bounds.0);
        assert!(self.exp <= bounds.1);
        let max_mantissa = MantissaTy::one_hot(Self::get_precision() as usize);
        assert!(self.mantissa.lt(&max_mantissa));
    }

    pub fn shift_significand_left(&mut self, amt: u64) {
        self.exp -= amt as i64;
        self.mantissa.shift_left(amt as usize);
    }

    pub fn shift_significand_right(&mut self, amt: u64) -> LossFraction {
        self.exp += amt as i64;
        let res = shift_right_with_loss(self.mantissa, amt);
        self.mantissa = res.0;
        res.1
    }

    /// \returns True if we need to round away from zero (increment the mantissa).
    fn need_round_away_from_zero(
        &self,
        rm: RoundingMode,
        loss: LossFraction,
    ) -> bool {
        assert!(self.is_normal() || self.is_zero());
        match rm {
            RoundingMode::Positive => !self.sign,
            RoundingMode::Negative => self.sign,
            RoundingMode::Zero => false,
            RoundingMode::NearestTiesToAway => loss.is_gte_half(),
            RoundingMode::NearestTiesToEven => {
                if loss.is_mt_half() {
                    return true;
                }

                loss.is_exactly_half() && self.mantissa.is_odd()
            }
        }
    }

    pub fn normalize(&mut self, rm: RoundingMode, loss: LossFraction) {
        if !self.is_normal() {
            return;
        }
        let mut loss = loss;
        let bounds = Self::get_exp_bounds();

        let nmsb = self.mantissa.msb_index() as i64;

        // Step I - adjust the exponent.
        if nmsb > 0 {
            // Align the number so that the MSB bit will be MANTISSA + 1.
            let mut exp_change = nmsb - Self::get_precision() as i64;

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
                assert!(loss.is_exactly_zero(), "losing information");
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
                *self = Self::zero(self.sign);
                return;
            }
            return;
        }

        // Check if we need to round away from zero.
        if self.need_round_away_from_zero(rm, loss) {
            if self.mantissa.is_zero() {
                self.exp = bounds.0
            }

            let one = MantissaTy::one();
            self.mantissa = self.mantissa + one;
            // Did the mantissa overflow?
            let mut m = self.mantissa;
            m.shift_right(Self::get_precision() as usize);
            if !m.is_zero() {
                // Can we fix the exponent?
                if self.exp < bounds.1 {
                    self.shift_significand_right(1);
                } else {
                    *self = Self::inf(self.sign);
                    return;
                }
            }
        }

        // Canonicalize.
        if self.mantissa.is_zero() {
            *self = Self::zero(self.sign);
        }
    } // round.
}

use std::cmp::Ordering;

/// Implement IEEE 754-2019 section 5.10 - totalOrder.
impl<const EXPONENT: usize, const MANTISSA: usize> PartialEq
    for Float<EXPONENT, MANTISSA>
{
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

/// Implement IEEE 754-2019 section 5.10 - totalOrder.
impl<const EXPONENT: usize, const MANTISSA: usize> PartialOrd
    for Float<EXPONENT, MANTISSA>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
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
                    bool_to_ord(!other.sign)
                } else if self.exp > other.exp {
                    bool_to_ord(self.sign)
                } else {
                    Some(self.mantissa.cmp(&other.mantissa))
                }
            }
        }
    }
}

#[test]
fn test_comparisons() {
    use super::utils;

    // Compare a bunch of special values, using the <,>,== operators and check
    // that they match the comparison on doubles.
    for first in utils::get_special_test_values() {
        for second in utils::get_special_test_values() {
            println!("{}  {}", first, second);
            let is_less = first < second;
            let is_eq = first == second;
            let is_gt = first > second;
            let first = FP64::from_f64(first);
            let second = FP64::from_f64(second);
            assert_eq!(is_less, first < second, "<");
            assert_eq!(is_eq, first == second, "==");
            assert_eq!(is_gt, first > second, ">");
        }
    }
}
