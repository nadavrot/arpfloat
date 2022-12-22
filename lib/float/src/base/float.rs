use super::utils;

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
    // The significand, including the implicit bit, aligned to the left.
    // Format [1xxxxxxx........]
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
                println!("FP[{} E={} M = 0x{:x}]", sign, self.exp, m);
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
        let exp_min: i64 = -Self::get_bias();
        // The highest value is 0xFFFE, because 0xFFFF is used for signaling.
        let exp_max: i64 = (1 << EXPONENT) - Self::get_bias() - 2;
        (exp_min, exp_max)
    }
}

pub type FP16 = Float<5, 10>;
pub type FP32 = Float<8, 23>;
pub type FP64 = Float<11, 52>;

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    pub fn clip(&mut self) {
        if self.mantissa == 0 {
            *self = Self::zero(self.sign);
        } else if self.exp > Self::get_exp_bounds().1 {
            *self = Self::inf(self.sign);
        }
        if self.exp < Self::get_exp_bounds().0 {
            *self = Self::zero(self.sign);
        }
    }

    /// Round the mantissa to \p num_bits of accuracy, using the \p mode
    /// rounding mode. Zero the rest of the mantissa is filled with zeros.
    /// This operation may overflow, in which case we update the exponent.
    //
    /// xxxxxxxxx becomes xxxxy0000, where y, could be rounded up.
    pub fn round(&mut self, num_bits: usize, mode: RoundingMode) {
        if !self.is_normal() {
            return;
        }

        assert!(num_bits < 64);
        let val = self.mantissa;
        let rest_bits = 64 - num_bits;
        let is_odd = ((val >> rest_bits) & 0x1) == 1;
        let bottom = val & utils::mask(rest_bits) as u64;
        let half = 1 << (rest_bits - 1) as u64;

        // Clear the lower part.
        let val = (val >> rest_bits) << rest_bits;

        match mode {
            RoundingMode::Zero => {
                self.mantissa = val;
            }
            RoundingMode::NearestTiesToEven => {
                if bottom > half || ((bottom == half) && is_odd) {
                    // If the next few bits are over the half point then round up.
                    // Or if the next few bits are exactly half, break the tie and go to even.
                    // This may overflow, so we'll need to adjust the exponent.
                    let r = val.overflowing_add(1 << rest_bits);
                    self.mantissa = r.0;
                    if r.1 {
                        self.exp += 1;
                    }
                } else {
                    self.mantissa = val;
                }
            }
            _ => {
                panic!("Unsupported rounding mode");
            }
        }
    }
}

#[test]
fn test_round() {
    let a = 0b100001000001010010110111101011001111001010110000000000000000000;
    let b = 0b100001000001010010110111101011001111001011000000000000000000000;
    let mut val = FP64::new(false, 0, a);
    val.round(44, RoundingMode::NearestTiesToEven);
    assert_eq!(val.get_mantissa(), b);

    let a = 0b101111;
    let b = 0b110000;
    let c = 0b100000;
    let mut val = FP64::new(false, 0, a);
    val.round(60, RoundingMode::NearestTiesToEven);
    assert_eq!(val.get_mantissa(), b);

    let mut val = FP64::new(false, 0, a);
    val.round(60, RoundingMode::Zero);
    assert_eq!(val.get_mantissa(), c);
}
