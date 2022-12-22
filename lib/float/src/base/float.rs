#[derive(Debug, Clone, Copy)]
pub enum RoundingMode {
    NearestTiesToEven,
    ToZero,
}

#[derive(Debug, Clone, Copy)]
enum LossFraction {
    ExactlyZero,  //0000000
    LessThanHalf, //0xxxxxx
    ExactlyHalf,  //1000000
    MoreThanHalf, //1xxxxxx
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
    pub fn new(
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
        Self::new(sign, 0, 0, Category::Zero)
    }

    /// \returns a new infinity float.
    pub fn inf(sign: bool) -> Self {
        Self::new(sign, 0, 0, Category::Infinity)
    }

    /// \returns a new NaN float.
    pub fn nan(sign: bool) -> Self {
        Self::new(sign, 0, 0, Category::NaN)
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
        return false;
    }

    /// \returns True if the Float is a +- NaN.
    pub fn is_nan(&self) -> bool {
        if let Category::NaN = self.category {
            return true;
        }
        return false;
    }

    /// \returns True if the Float is a +- NaN.
    pub fn is_zero(&self) -> bool {
        if let Category::Zero = self.category {
            return true;
        }
        return false;
    }

    pub fn is_normal(&self) -> bool {
        if let Category::Normal = self.category {
            return true;
        }
        return false;
    }
}
