use super::utils::{self, mask};

#[derive(Debug, Clone, Copy)]
pub enum RoundingMode {
    NearestTiesToEven,
    NearestTiesToAway,
    Zero,
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy)]
pub enum LossFraction {
    ExactlyZero,  //0000000
    LessThanHalf, //0xxxxxx
    ExactlyHalf,  //1000000
    MoreThanHalf, //1xxxxxx
}

impl LossFraction {
    pub fn is_exactly_zero(&self) -> bool {
        matches!(self, Self::ExactlyZero)
    }
    pub fn is_lt_half(&self) -> bool {
        matches!(self, Self::LessThanHalf)
    }
    pub fn is_exactly_half(&self) -> bool {
        matches!(self, Self::ExactlyHalf)
    }
    pub fn is_mt_half(&self) -> bool {
        matches!(self, Self::MoreThanHalf)
    }
    pub fn is_lte_half(&self) -> bool {
        self.is_lt_half() || self.is_exactly_half()
    }
    pub fn is_gte_half(&self) -> bool {
        self.is_mt_half() || self.is_exactly_half()
    }

    // Return the inverted loss fraction.
    pub fn invert(&self) -> LossFraction {
        match self {
            LossFraction::LessThanHalf => LossFraction::MoreThanHalf,
            LossFraction::MoreThanHalf => LossFraction::LessThanHalf,
            _ => {*self}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Category {
    Infinity,
    NaN,
    Normal,
    Zero,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Float<const EXPONENT: usize, const MANTISSA: usize> {
    // The Sign bit.
    sign: bool,
    // The Exponent.
    exp: i64,
    // The significand, including the implicit bit, aligned to the right.
    // Format [00000001xxxxxxx].
    mantissa: u64,
    // The kind of number this float represents.
    category: Category,
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Create a new normal floating point number.
    pub fn new(sign: bool, exp: i64, mantissa: u64) -> Self {
        if mantissa == 0 {
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
        mantissa: u64,
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
            mantissa: 0,
            category: Category::Zero,
        }
    }

    /// \returns a new infinity float.
    pub fn inf(sign: bool) -> Self {
        Float {
            sign,
            exp: 0,
            mantissa: 0,
            category: Category::Infinity,
        }
    }

    /// \returns a new NaN float.
    pub fn nan(sign: bool) -> Self {
        Float {
            sign,
            exp: 0,
            mantissa: 0,
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

    pub fn get_mantissa(&self) -> u64 {
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
        if self.exp < other.get_exp() {
            true
        } else if self.exp > other.get_exp() {
            return false;
        } else {
            return self.mantissa < other.get_mantissa();
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
                println!("FP[{} E={} M = 0x{:64b}]", sign, self.exp, m);
            }
        }
    }

    /// Returns the exponent bias for the number, as a positive number.
    /// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
    pub fn get_bias() -> i64 {
        utils::compute_ieee745_bias(EXPONENT) as i64
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

/// \returns the fractional part that's lost during truncation of  \p val.
/// Val needs to be aligned to the left, so the MSB of the number is aligned
/// to the MSB of the word.
pub fn get_loss_kind_of_trunc(val: u64) -> LossFraction {
    if val == 0 {
        return LossFraction::ExactlyZero;
    } else if val == (1 << 63) {
        return LossFraction::ExactlyHalf;
    } else if val > (1 << 63) {
        return LossFraction::MoreThanHalf;
    }
    LossFraction::LessThanHalf
}

//// Shift \p val by \p bits, and report the loss.
fn shift_right_with_loss(val: u64, bits: u64) -> (u64, LossFraction) {
    if bits < 64 {
        let loss = get_loss_kind_of_trunc(val << (64 - bits));
        (val >> bits, loss)
    } else {
        let loss = get_loss_kind_of_trunc(val);
        (0, loss)
    }
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
    let res = shift_right_with_loss(0b10000000, 3);
    assert!(res.1.is_exactly_zero());

    let res = shift_right_with_loss(0b10000111, 3);
    assert!(res.1.is_mt_half());

    let res = shift_right_with_loss(0b10000100, 3);
    assert!(res.1.is_exactly_half());

    let res = shift_right_with_loss(0b10000001, 3);
    assert!(res.1.is_lt_half());
}

/// \returns the first digit after the msb. This allows us to support
/// MSB index of zero.
fn next_msb(val: u64) -> u64 {
    64 - val.leading_zeros() as u64
}

#[test]
fn text_next_msb() {
    assert_eq!(next_msb(0x0), 0);
    assert_eq!(next_msb(0x1), 1);
    assert_eq!(next_msb(0xff), 8);
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    fn overflow(&mut self, rm: RoundingMode) {
        let bounds = Self::get_exp_bounds();
        let inf = Self::inf(self.sign);
        let max = Self::new(self.sign, bounds.1, mask(MANTISSA) as u64);

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

    fn check_bounds(&self) {
        let bounds = Self::get_exp_bounds();
        assert!(self.exp >= bounds.0);
        assert!(self.exp <= bounds.1);
        assert!(self.mantissa <= 1 << Self::get_precision());
    }

    pub fn shift_significand_left(&mut self, amt: u64) {
        self.exp -= amt as i64;
        self.mantissa <<= amt;
        self.check_bounds()
    }

    pub fn shift_significand_right(&mut self, amt: u64) -> LossFraction {
        self.exp += amt as i64;
        let res = shift_right_with_loss(self.mantissa, amt);
        self.mantissa = res.0;
        self.check_bounds();
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

                loss.is_exactly_half() && self.mantissa & 0x1 == 1
            }
        }
    }

    pub fn normalize(&mut self, rm: RoundingMode, loss: LossFraction) {
        if !self.is_normal() {
            return;
        }
        let mut loss = loss;
        let bounds = Self::get_exp_bounds();

        let nmsb = next_msb(self.mantissa) as i64;

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

            // Handle underflowing low exponents.
            if self.exp + exp_change < bounds.0 {
                // TODO: we ignore denormal encoding here and pretend that they
                // don't exist in normalized floats. Should the float/double encoder
                // handle them?
                exp_change = bounds.0 - self.exp;
            }

            if exp_change < 0 {
                // Handle reducing the exponent.
                assert!(loss.is_exactly_zero(), "losing information");
                self.shift_significand_left(-exp_change as u64);
                return;
            } else if exp_change > 0 {
                // Handle increasing the exponent.
                let loss2 = self.shift_significand_right(exp_change as u64);
                loss = combine_loss_fraction(loss2, loss);
            }
        }

        //Step II - round the number.

        // If nothing moved or the shift didn't mess things up then we're done.
        if loss.is_exactly_zero() {
            // Canonicalize to zero.
            if self.mantissa == 0 {
                *self = Self::zero(self.sign);
                return;
            }
            return;
        }

        // Check if we need to round away from zero.
        if self.need_round_away_from_zero(rm, loss) {
            if self.mantissa == 0 {
                self.exp = bounds.0
            }

            self.mantissa += 1;
            // Did the mantissa overflow?
            if (self.mantissa >> Self::get_precision()) > 0 {
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
        if self.mantissa == 0 {
            *self = Self::zero(self.sign);
        }
    } // round.
}
