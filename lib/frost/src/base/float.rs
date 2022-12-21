use super::utils;

use super::utils::{mask, RoundMode};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Float<const EXPONENT: usize, const MANTISSA: usize> {
    // The Sign bit.
    sign: bool,
    // The Exponent.
    exp: u64,
    // The significand, including the possible implicit bit, aligned to the
    // left. Format [1xxxxxxx........]
    mantissa: u64,
}

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    pub fn new(sign: bool, exp: i64, sig: u64) -> Self {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_exp(exp);
        a.set_mantissa(sig);
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
    fn get_unbiased_exp(&self) -> u64 {
        self.exp
    }

    /// \returns the biased exponent.
    pub fn get_exp(&self) -> i64 {
        self.exp as i64 - Self::get_bias() as i64
    }

    /// Sets the biased exponent to \p new_exp.
    pub fn set_exp(&mut self, new_exp: i64) {
        let (exp_min, exp_max) = Self::get_exp_bounds();
        assert!(new_exp <= exp_max);
        assert!(exp_min <= new_exp);

        let new_exp: i64 = new_exp + (Self::get_bias() as i64);
        self.exp = new_exp as u64
    }

    /// Sets the unbiased exponent to \p new_exp.
    fn set_unbiased_exp(&mut self, new_exp: u64) {
        self.exp = new_exp
    }

    pub fn get_bias() -> u64 {
        utils::compute_ieee745_bias(EXPONENT) as u64
    }

    /// \returns the bounds of the upper and lower bounds of the exponent.
    pub fn get_exp_bounds() -> (i64, i64) {
        let exp_min: i64 = -(Self::get_bias() as i64);
        // The highest value is 0xFFFE, because 0xFFFF is used for signaling.
        let exp_max: i64 = ((1 << EXPONENT) - Self::get_bias() - 2) as i64;
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
        println!(
            "FP[S={} : E={} (biased {}) :SI=0x{:x}]",
            sign, self.exp, exp, mantissa
        );
    }

    /// Convert denormals into normal values. Notice that this
    /// may create exponent values that are not legal.
    pub fn normalize(&mut self) {
        if self.is_normal() || self.is_zero() {
            return;
        }

        let lz = self.mantissa.leading_zeros() as u64;
        assert!(lz > 0 && lz < 64);
        self.mantissa <<= lz;
        self.exp -= lz;
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

    let a: Float<6, 10> = Float::new(false, 2, 12);
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
