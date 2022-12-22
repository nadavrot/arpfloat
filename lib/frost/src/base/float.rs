use super::utils;

use super::utils::{mask, RoundMode};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Float<const EXPONENT: usize, const MANTISSA: usize> {
    // The Sign bit.
    sign: bool,
    // The Exponent.
    exp: i64,
    // The significand, including the possible implicit bit, aligned to the
    // left. Format [1xxxxxxx........]
    mantissa: u64,
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Create a new normal floating point number.
    pub fn new(sign: bool, exp: i64, m: u64) -> Self {
        assert!(m.leading_zeros() == 0, "A normal Mantissa starts w/ 1 msb");
        assert_ne!(exp, Self::get_exp_bounds().0);
        Self::raw(sign, exp, m)
    }
    /// Create a new normal floating point number, without checks.
    pub fn raw(sign: bool, exp: i64, m: u64) -> Self {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_exp(exp);
        a.set_mantissa(m);
        a
    }

    /// Create a new de-normal floating point number.
    pub fn new_denormal(sign: bool, m: u64) -> Self {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_unbiased_exp(0);
        a.set_mantissa(m);
        a
    }

    pub fn zero(sign: bool) -> Self {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_unbiased_exp(0);
        a.set_mantissa(0);
        a
    }

    pub fn inf(sign: bool) -> Self {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_unbiased_exp(mask(EXPONENT) as u64);
        a.set_mantissa(0);
        a
    }
    pub fn nan(sign: bool) -> Self {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_unbiased_exp(mask(EXPONENT) as u64);
        // Set two bits: the implicit mantissa 1' and a non-zero payload.
        // [1.1000.....]
        a.set_mantissa(0xc000_0000_0000_0000);
        a
    }

    /// \returns True if the Float has the signaling exponent.
    pub fn in_special_exp(&self) -> bool {
        self.get_unbiased_exp() == mask(EXPONENT) as u64
    }

    /// \returns True if the Float is negative
    pub fn is_negative(&self) -> bool {
        self.get_sign()
    }

    /// \returns True if the Float is a positive or negative infinity.
    pub fn is_inf(&self) -> bool {
        self.in_special_exp() && self.get_frac_mantissa() == 0
    }

    /// \returns True if the Float is a positive or negative NaN.
    pub fn is_nan(&self) -> bool {
        self.in_special_exp() && self.get_frac_mantissa() != 0
    }

    /// \returns True if the Float is a positive or negative NaN.
    pub fn is_zero(&self) -> bool {
        self.get_unbiased_exp() == 0 && self.get_mantissa() == 0
    }

    pub fn is_normal(&self) -> bool {
        self.get_unbiased_exp() != 0
    }

    /// \returns the sign bit.
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    /// Sets the sign to \p s.
    pub fn set_sign(&mut self, s: bool) {
        self.sign = s;
    }

    /// \returns the mantissa (including the implicit 0/1 bit).
    pub fn get_mantissa(&self) -> u64 {
        self.mantissa
    }

    /// \return the fractional part of the mantissa without the implicit 1 or 0.
    /// [(0/1).xxxxxx].
    fn get_frac_mantissa(&self) -> u64 {
        self.mantissa << 1
    }

    /// Sets the mantissa to \p sg (including the implicit 0/1 bit).
    pub fn set_mantissa(&mut self, sg: u64) {
        self.mantissa = sg;
    }

    /// \returns the unbiased exponent.
    pub fn get_unbiased_exp(&self) -> u64 {
        (self.exp + Self::get_bias()) as u64
    }

    /// \returns the biased exponent.
    pub fn get_exp(&self) -> i64 {
        self.exp
    }

    /// Sets the biased exponent to \p new_exp.
    pub fn set_exp(&mut self, new_exp: i64) {
        self.exp = new_exp
    }

    /// Sets the unbiased exponent to \p new_exp.
    fn set_unbiased_exp(&mut self, new_exp: u64) {
        self.exp = new_exp as i64 - Self::get_bias()
    }

    /// Returns the exponent bias for the number, as a positive number.
    /// https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
    fn get_bias() -> i64 {
        utils::compute_ieee745_bias(EXPONENT) as i64
    }

    /// \returns the bounds of the upper and lower bounds of the exponent.
    pub fn get_exp_bounds() -> (i64, i64) {
        let exp_min: i64 = -Self::get_bias();
        // The highest value is 0xFFFE, because 0xFFFF is used for signaling.
        let exp_max: i64 = (1 << EXPONENT) - Self::get_bias() - 2;
        (exp_min, exp_max)
    }

    pub fn neg(&self) -> Self {
        let mut a = *self;
        a.set_sign(!self.get_sign());
        a
    }

    pub fn dump(&self) {
        let exp = self.get_exp();
        let mantissa = self.get_mantissa();
        let sign = self.get_sign() as usize;
        let dn = if self.is_normal() { "" } else { "*" };
        println!("FP[S={} : E={} {}: M = 0x{:x}]", sign, exp, dn, mantissa);
    }
}

pub type FP16 = Float<5, 10>;
pub type FP32 = Float<8, 23>;
pub type FP64 = Float<11, 52>;

#[test]
fn setter_test() {
    assert_eq!(FP16::get_bias(), 15);
    assert_eq!(FP32::get_bias(), 127);
    assert_eq!(FP64::get_bias(), 1023);

    let a: Float<6, 10> = Float::new(false, 2, (1 << 63) + 12);
    let mut b = a;
    b.set_exp(b.get_exp());
    assert_eq!(a.get_exp(), b.get_exp());
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Round the mantissa to \p num_bits of accuracy, using the \p mode
    /// rounding mode. Zero the rest of the mantissa is filled with zeros.
    /// This operation may overflow, in which case we update the exponent.
    //
    /// xxxxxxxxx becomes xxxxy0000, where y, could be rounded up.
    pub fn round(&mut self, num_bits: usize, mode: RoundMode) {
        assert!(num_bits < 64);
        let val = self.mantissa;
        let rest_bits = 64 - num_bits;
        let is_odd = ((val >> rest_bits) & 0x1) == 1;
        let bottom = val & mask(rest_bits) as u64;
        let half = 1 << (rest_bits - 1) as u64;

        // Clear the lower part.
        let val = (val >> rest_bits) << rest_bits;

        match mode {
            RoundMode::Trunc => {
                self.mantissa = val;
            }
            RoundMode::Even => {
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
        }
    }
}

#[test]
fn test_round() {
    let a = 0b100001000001010010110111101011001111001010110000000000000000000;
    let b = 0b100001000001010010110111101011001111001011000000000000000000000;
    let mut val = FP64::default();
    val.set_mantissa(a);
    val.round(44, RoundMode::Even);
    assert_eq!(val.get_mantissa(), b);

    let a = 0b101111;
    let b = 0b110000;
    let c = 0b100000;
    let mut val = FP64::default();
    val.set_mantissa(a);
    val.round(60, RoundMode::Even);
    assert_eq!(val.get_mantissa(), b);

    let mut val = FP64::default();
    val.set_mantissa(a);
    val.round(60, RoundMode::Trunc);
    assert_eq!(val.get_mantissa(), c);
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// \return true if the number is in a legal form, which means that it is
    /// normalized and within the allowed exponent range.
    pub fn is_legal(&self) -> bool {
        let bounds = Self::get_exp_bounds();
        let is1x = self.mantissa.leading_zeros() == 0;
        // Check if this is a number in the normal range.
        if is1x && (self.exp <= bounds.1 && self.exp > bounds.0) {
            true
        } else if self.exp == bounds.0 && !is1x {
            // This is a denormal number.
            true
        } else {
            // Nan, Inf are legal.
            self.in_special_exp()
        }
    }

    /// Convert the number into a normal form: denormal, normal, zero or
    /// infinity. The procedure handles normals that need to turn into
    /// denormals, and numbers that fall out of bounds.
    pub fn legalize(&mut self) {
        if self.is_legal() {
            return;
        }

        let bounds = Self::get_exp_bounds();

        // Normalize zeros.
        if self.mantissa == 0 {
            *self = Self::zero(self.sign);
            return;
        }

        // Try to promote denormals.
        let lz = self.mantissa.leading_zeros();
        if lz != 0 && ((self.exp + lz as i64) > bounds.0) {
            // This is a mantissa without a leading 1 that does not have a
            // denormal exponent. Shift it into the normal range and legalize
            // the exponent later.
            self.mantissa <<= lz;
            self.exp -= lz as i64;
            assert_ne!(self.exp, bounds.0);
        }

        // If the number should have an explicit leading 1 if it's not a
        // denormal.
        assert!(!(lz == 0 && self.exp == bounds.0));

        // Handle small exponents.
        if self.exp < bounds.0 {
            // The exponent is below the allowed exponent range. Try to turn it
            // into a denormal.
            let exp_delta = bounds.0 - self.exp;
            if exp_delta > 0 && exp_delta < 64 {
                self.mantissa >>= exp_delta + 1;
                self.exp = bounds.0;
                return;
            }
            // The number is too small. Drop to zero.
            *self = Self::zero(self.sign);
            return;
        }

        // Handle large exponents.
        if self.exp > bounds.1 {
            *self = Self::inf(self.sign);
        }
    }
}

#[test]
fn test_legalization() {
    // Small number to zero.
    let mut a = FP32::raw(false, -1000, 0x0000000013371337);
    a.legalize();
    assert!(a.is_legal() && a.is_zero() && !a.is_inf() && !a.is_normal());

    // Small normal number to a small normal number.
    let mut a = FP32::raw(false, 0, 0x1);
    a.legalize();
    assert!(a.is_legal());
    assert_eq!(a.get_exp(), -63);

    // Denorm to denorm.
    let mut a = FP32::raw(false, -120, 0x1);
    a.legalize();
    assert!(a.is_legal());
    assert_eq!(a.get_exp(), -127);
    assert_eq!(a.get_mantissa(), 64);

    // Denorm to a small normal number.
    let mut a = FP32::raw(false, -120, 1 << 60);
    a.legalize();
    assert!(a.is_legal());
    assert_eq!(a.get_exp(), -123);
    assert_eq!(a.get_mantissa(), 0x8000000000000000);
}
