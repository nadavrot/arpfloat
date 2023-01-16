//! This module contains the implementation of casting-related methods.

use crate::float::Semantics;
use crate::FP128;

use super::bigint::BigInt;
use super::bigint::LossFraction;
use super::float::{self, Category};
use super::float::{Float, RoundingMode, FP32, FP64};
use super::utils;
use super::utils::mask;

impl Float {
    /// Load the integer `val` into the float. Notice that the number may
    /// overflow, or rounded to the nearest even integer.
    pub fn from_u64(sem: Semantics, val: u64) -> Self {
        Self::from_bigint(FP128, BigInt::from_u64(val)).cast(sem)
    }

    /// Load the big int `val` into the float. Notice that the number may
    /// overflow, or rounded to the nearest even integer.
    pub fn from_bigint(sem: Semantics, val: BigInt) -> Self {
        let mut a = Self::new(sem, false, sem.get_mantissa_len() as i64, val);
        a.normalize(sem.get_rounding_mode(), LossFraction::ExactlyZero);
        a
    }

    /// Load the integer `val` into the float. Notice that the number may
    /// overflow or rounded.
    pub fn from_i64(sem: Semantics, val: i64) -> Self {
        if val < 0 {
            let mut a = Self::from_u64(sem, -val as u64);
            a.set_sign(true);
            return a;
        }

        Self::from_u64(sem, val as u64)
    }

    /// Converts and returns the rounded integral part.
    pub fn to_i64(&self) -> i64 {
        if self.is_nan() || self.is_zero() {
            return 0;
        }

        if self.is_inf() {
            if self.get_sign() {
                return i64::MIN;
            } else {
                return i64::MAX;
            }
        }
        let rm = self.get_semantics().get_rounding_mode();
        let val = self.convert_normal_to_integer(rm);
        if self.get_sign() {
            -(val.as_u64() as i64)
        } else {
            val.as_u64() as i64
        }
    }

    /// Returns a value that is rounded to the nearest integer that's not larger
    /// in magnitude than this float.
    pub fn trunc(&self) -> Self {
        // Only handle normal numbers (don't do anything to NaN, Inf, Zero).
        if !self.is_normal() {
            return self.clone();
        }

        let exp = self.get_exp();

        if exp > self.get_mantissa_len() as i64 {
            // Already an integer.
            return self.clone();
        }

        // Numbers that are smaller than 1 are rounded to zero.
        if exp < -1 {
            return Self::zero(self.get_semantics(), self.get_sign());
        }

        // This is a fraction. Figure out which bits represent values over one
        // and clear out the values that represent the fraction.
        let trim = (self.get_mantissa_len() as i64 - exp) as usize;
        let mut m = self.get_mantissa();
        m.shift_right(trim);
        m.shift_left(trim);
        Self::new(self.get_semantics(), self.get_sign(), self.get_exp(), m)
    }

    /// Returns a number rounded to nearest integer, away from zero.
    pub fn round(&self) -> Self {
        use crate::float::shift_right_with_loss;
        let sem = self.get_semantics();

        // Only handle normal numbers (don't do anything to NaN, Inf, Zero).
        if !self.is_normal() {
            return self.clone();
        }

        let exp = self.get_exp();

        if exp > self.get_mantissa_len() as i64 {
            // Already an integer.
            return self.clone();
        }

        // Numbers that are between 0.5 and 1.0 are rounded to 1.0.
        if exp == -1 {
            return Self::one(sem, self.get_sign());
        }

        // Numbers below 0.5 are rounded to zero.
        if exp < -2 {
            return Self::zero(sem, self.get_sign());
        }

        // This is a fraction. Figure out which bits represent values over one
        // and clear out the values that represent the fraction.
        let trim = (self.get_mantissa_len() as i64 - exp) as usize;
        let (mut m, loss) = shift_right_with_loss(&self.get_mantissa(), trim);
        m.shift_left(trim);
        let t = Self::new(sem, self.get_sign(), self.get_exp(), m);

        if loss.is_lt_half() {
            t
        } else if self.get_sign() {
            t - Self::one(sem, false)
        } else {
            t + Self::one(sem, false)
        }
    }

    pub(crate) fn convert_normal_to_integer(&self, rm: RoundingMode) -> BigInt {
        // We are converting to integer, so set the center point of the exponent
        // to the lsb instead of the msb.
        let i_exp = self.get_exp() - self.get_mantissa_len() as i64;
        if i_exp < 0 {
            let (mut m, loss) = float::shift_right_with_loss(
                &self.get_mantissa(),
                -i_exp as usize,
            );
            if self.need_round_away_from_zero(rm, loss) {
                m.inplace_add(&BigInt::one());
            }
            m
        } else {
            let mut m = self.get_mantissa();
            m.shift_left(i_exp as usize);
            m
        }
    }

    fn from_bits(sem: Semantics, float: u64) -> Self {
        // Extract the biased exponent (wipe the sign and mantissa).
        let biased_exp = ((float >> sem.get_mantissa_len())
            & mask(sem.get_exponent_len()) as u64)
            as i64;
        // Wipe the original exponent and mantissa.
        let sign =
            (float >> (sem.get_exponent_len() + sem.get_mantissa_len())) & 1;
        // Wipe the sign and exponent.
        let mut mantissa = float & mask(sem.get_mantissa_len()) as u64;

        let sign = sign == 1;

        // Check for NaN/Inf
        if biased_exp == mask(sem.get_exponent_len()) as i64 {
            if mantissa == 0 {
                return Self::inf(sem, sign);
            }
            return Self::nan(sem, sign);
        }

        let mut exp = biased_exp - sem.get_bias();

        // Add the implicit bit for normal numbers.
        if biased_exp != 0 {
            mantissa += 1u64 << sem.get_mantissa_len();
        } else {
            // Handle denormals, adjust the exponent to the legal range.
            exp += 1;
        }

        let mantissa = BigInt::from_u64(mantissa);
        Self::new(sem, sign, exp, mantissa)
    }

    /// Cast to another float using the non-default rounding mode `rm`.
    pub fn cast_with_rm(&self, to: Semantics, rm: RoundingMode) -> Float {
        let mut loss = LossFraction::ExactlyZero;
        let exp_delta =
            self.get_mantissa_len() as i64 - to.get_mantissa_len() as i64;
        let mut temp = self.clone();
        // If we are casting to a narrow type then we need to shift the bits
        // to the new-mantissa part of the word. This will adjust the exponent,
        // and if we lose bits then we'll need to round the number.
        if exp_delta > 0 {
            loss = temp.shift_significand_right(exp_delta as u64);
        }

        let mut x = Float::raw(
            to,
            temp.get_sign(),
            temp.get_exp() - exp_delta,
            temp.get_mantissa(),
            temp.get_category(),
        );
        // Don't normalize if this is a nop conversion.
        if to.get_exponent_len() != self.get_exponent_len()
            || to.get_mantissa_len() != self.get_mantissa_len()
        {
            x.normalize(rm, loss);
        }
        x
    }
    /// Convert from one float format to another.
    pub fn cast(&self, to: Semantics) -> Float {
        self.cast_with_rm(to, self.get_semantics().get_rounding_mode())
    }

    fn as_native_float(&self) -> u64 {
        // https://en.wikipedia.org/wiki/IEEE_754
        let mantissa: u64;
        let mut exp: u64;
        match self.get_category() {
            Category::Infinity => {
                mantissa = 0;
                exp = mask(self.get_exponent_len()) as u64;
            }
            Category::NaN => {
                mantissa = 1 << (self.get_mantissa_len() - 1);
                exp = mask(self.get_exponent_len()) as u64;
            }
            Category::Zero => {
                mantissa = 0;
                exp = 0;
            }
            Category::Normal => {
                exp = (self.get_exp() + self.get_bias()) as u64;
                debug_assert!(exp > 0);
                let m = self.get_mantissa().as_u64();
                // Encode denormals. If the exponent is the minimum value and we
                // don't have a leading integer bit (in the form 1.mmmm) then
                // this is a denormal value and we need to encode it as such.
                if (exp == 1) && ((m >> self.get_mantissa_len()) == 0) {
                    exp = 0;
                }
                mantissa = m & utils::mask(self.get_mantissa_len()) as u64;
            }
        }

        let mut bits: u64 = self.get_sign() as u64;
        bits <<= self.get_exponent_len();
        bits |= exp;
        bits <<= self.get_mantissa_len();
        debug_assert!(mantissa <= 1 << self.get_mantissa_len());
        bits |= mantissa;
        bits
    }
    /// Convert this float to fp32. Notice that the number may overflow or
    /// rounded to the nearest even (see cast and cast_with_rm).
    pub fn as_f32(&self) -> f32 {
        let b = self.cast(FP32);
        let bits = b.as_native_float();
        f32::from_bits(bits as u32)
    }
    /// Convert this float to fp64. Notice that the number may overflow or
    /// rounded to the nearest even (see cast and cast_with_rm).
    pub fn as_f64(&self) -> f64 {
        let b = self.cast(FP64);
        let bits = b.as_native_float();
        f64::from_bits(bits)
    }

    /// Loads and converts a native fp32 value. Notice that the number may
    /// overflow or rounded (see cast and cast_with_rm).
    pub fn from_f32(float: f32) -> Self {
        Float::from_bits(FP32, float.to_bits() as u64)
    }

    /// Loads and converts a native fp64 value. Notice that the number may
    /// overflow or rounded (see cast and cast_with_rm).
    pub fn from_f64(float: f64) -> Self {
        Float::from_bits(FP64, float.to_bits())
    }
}

#[test]
fn test_rounding_to_integer() {
    // Test the low integers with round-to-zero.
    for i in 0..100 {
        let z64 = FP64.with_rm(RoundingMode::Zero);
        let r = Float::from_f64(i as f64 + 0.1).cast(z64).to_i64();
        assert_eq!(i, r);
    }

    // Test the high integers with round_to_zero.
    for i in 0..100 {
        let z64 = FP64.with_rm(RoundingMode::Zero);
        let val = (i as i64) << 54;
        let r = Float::from_i64(FP64, val).cast(z64).to_i64();
        assert_eq!(val, r);
    }

    let nta64 = FP64.with_rm(RoundingMode::NearestTiesToAway);
    assert_eq!(1, Float::from_f64(0.5).cast(nta64).to_i64());
    assert_eq!(0, Float::from_f64(0.49).cast(nta64).to_i64());
    assert_eq!(199999, Float::from_f64(199999.49).cast(nta64).to_i64());
    assert_eq!(0, Float::from_f64(-0.49).cast(nta64).to_i64());
    assert_eq!(-1, Float::from_f64(-0.5).cast(nta64).to_i64());

    let z64 = FP64.with_rm(RoundingMode::Zero);
    assert_eq!(0, Float::from_f64(0.9).cast(z64).to_i64());
    assert_eq!(1, Float::from_f64(1.1).cast(z64).to_i64());
    assert_eq!(99, Float::from_f64(99.999).cast(z64).to_i64());
    assert_eq!(0, Float::from_f64(-0.99).cast(z64).to_i64());
    assert_eq!(0, Float::from_f64(-0.5).cast(z64).to_i64());

    let p64 = FP64.with_rm(RoundingMode::Positive);
    assert_eq!(1, Float::from_f64(0.9).cast(p64).to_i64());
    assert_eq!(2, Float::from_f64(1.1).cast(p64).to_i64());
    assert_eq!(100, Float::from_f64(99.999).cast(p64).to_i64());
    assert_eq!(0, Float::from_f64(-0.99).cast(p64).to_i64());
    assert_eq!(0, Float::from_f64(-0.5).cast(p64).to_i64());

    // Special values
    let n_inf = f64::NEG_INFINITY;
    let inf = f64::INFINITY;
    assert_eq!(0, Float::from_f64(f64::NAN).to_i64());
    assert_eq!(i64::MIN, Float::from_f64(n_inf).to_i64());
    assert_eq!(i64::MAX, Float::from_f64(inf).to_i64());
}

#[test]
fn test_round_trip_native_float_cast() {
    let f = f32::from_bits(0x41700000);
    let a = Float::from_f32(f);
    assert_eq!(f, a.as_f32());

    let pi = 355. / 113.;
    let a = Float::from_f64(pi);
    assert_eq!(pi, a.as_f64());

    assert!(Float::from_f64(f64::NAN).is_nan());
    assert!(!Float::from_f64(f64::NAN).is_inf());
    assert!(Float::from_f64(f64::INFINITY).is_inf());
    assert!(!Float::from_f64(f64::INFINITY).is_nan());
    assert!(Float::from_f64(f64::NEG_INFINITY).is_inf());

    let a_float = f32::from_bits(0x3f8fffff);
    let a = Float::from_f32(a_float);
    let b = a.cast(FP32);
    assert_eq!(a.as_f32(), a_float);
    assert_eq!(b.as_f32(), a_float);

    let f = f32::from_bits(0x000000);
    let a = Float::from_f32(f);
    assert!(!a.is_normal());
    assert_eq!(f, a.as_f32());
}

#[test]
fn test_cast_easy_ctor() {
    let values = [0x3f8fffff, 0x40800000, 0x3f000000, 0xc60b40ec, 0xbc675793];

    for v in values {
        let output = f32::from_bits(v);
        let a = Float::from_f32(output).cast(FP64);
        let b = a.cast(FP32);
        assert_eq!(a.as_f32(), output);
        assert_eq!(b.as_f32(), output);
    }
}

#[test]
fn test_cast_from_integers() {
    use super::float::FP16;

    let pi = 355. / 133.;
    let e = 193. / 71.;

    assert_eq!(Float::from_i64(FP32, 1 << 32).as_f32(), (1u64 << 32) as f32);
    assert_eq!(Float::from_i64(FP32, 1 << 34).as_f32(), (1u64 << 34) as f32);
    assert_eq!(Float::from_f64(pi).as_f32(), (pi) as f32);
    assert_eq!(Float::from_f64(e).as_f32(), (e) as f32);
    assert_eq!(Float::from_u64(FP32, 8388610).as_f32(), 8388610 as f32);

    for i in 0..(1 << 16) {
        assert_eq!(Float::from_u64(FP32, i << 12).as_f32(), (i << 12) as f32);
    }

    assert_eq!(Float::from_i64(FP16, 0).as_f64(), 0.);
    assert_eq!(Float::from_i64(FP16, 65500).as_f64(), 65504.0);
    assert_eq!(Float::from_i64(FP16, 65504).as_f64(), 65504.0);
    assert_eq!(Float::from_i64(FP16, 65519).as_f64(), 65504.0);
    assert_eq!(Float::from_i64(FP16, 65520).as_f64(), f64::INFINITY);
    assert_eq!(Float::from_i64(FP16, 65536).as_f64(), f64::INFINITY);

    for i in -100..100 {
        let a = Float::from_i64(FP32, i);
        let b = Float::from_f64(i as f64).cast(FP32);
        assert_eq!(a.as_f32(), b.as_f32());
    }
}

#[test]
fn test_cast_zero_nan_inf() {
    assert!(Float::nan(FP64, true).as_f64().is_nan());
    assert_eq!(Float::zero(FP64, false).as_f64(), 0.0);
    assert_eq!(Float::zero(FP64, true).as_f64(), -0.0);

    assert!(Float::nan(FP64, true).is_nan());
    assert!(Float::inf(FP64, true).is_inf());
    {
        let a = Float::from_f32(f32::from_bits(0x3f8fffff));
        assert!(!a.is_inf());
        assert!(!a.is_nan());
        assert!(!a.is_negative());
    }
    {
        let a = Float::from_f32(f32::from_bits(0xf48fffff));
        assert!(!a.is_inf());
        assert!(!a.is_nan());
        assert!(a.is_negative());
    }
    {
        let a = Float::from_f32(f32::from_bits(0xff800000)); // -Inf
        assert!(a.is_inf());
        assert!(!a.is_nan());
        assert!(a.is_negative());
    }
    {
        let a = Float::from_f32(f32::from_bits(0xffc00000)); // -Nan.
        assert!(!a.is_inf());
        assert!(a.is_nan());
        assert!(a.is_negative());
    }

    {
        let a = Float::from_f64(f64::from_bits((mask(32) << 32) as u64));
        assert!(!a.is_inf());
        assert!(a.is_nan());
    }
    {
        // Check that casting propagates inf/nan.
        let a = Float::from_f32(f32::from_bits(0xff800000)); // -Inf
        let b = a.cast(FP64);
        assert!(b.is_inf());
        assert!(!b.is_nan());
        assert!(b.is_negative());
    }
}

#[test]
fn test_cast_down_easy() {
    // Check that we can cast the numbers down, matching the hardware casting.
    for v in [0.3, 0.1, 14151241515., 14151215., 0.0000000001, 1000000000.] {
        let res = Float::from_f64(v).as_f32();
        assert_eq!(Float::from_f64(v).as_f64().to_bits(), v.to_bits());
        assert!(res == v as f32);
    }
}

#[test]
fn test_load_store_all_f32() {
    // Try to load and store normals and denormals.
    for i in 0..(1u64 << 16) {
        let in_f = f32::from_bits((i << 10) as u32);
        let fp_f = Float::from_f32(in_f);
        let out_f = fp_f.as_f32();
        assert_eq!(in_f.is_nan(), out_f.is_nan());
        assert_eq!(in_f.is_infinite(), out_f.is_infinite());
        assert!(in_f.is_nan() || (in_f.to_bits() == out_f.to_bits()));
    }
}

#[cfg(feature = "std")]
#[test]
fn test_cast_down_complex() {
    // Try casting a bunch of difficult values such as inf, nan, denormals, etc.
    for v in utils::get_special_test_values() {
        let res = Float::from_f64(v).as_f32();
        assert_eq!(Float::from_f64(v).as_f64().to_bits(), v.to_bits());
        assert_eq!(v.is_nan(), res.is_nan());
        assert!(v.is_nan() || res == v as f32);
    }
}

#[cfg(feature = "std")]
#[test]
fn test_trunc() {
    use super::utils::Lfsr;

    let large_integer = (1u64 << 52) as f64;
    assert_eq!(Float::from_f64(0.4).trunc().as_f64(), 0.);
    assert_eq!(Float::from_f64(1.4).trunc().as_f64(), 1.);
    assert_eq!(Float::from_f64(1.99).trunc().as_f64(), 1.);
    assert_eq!(Float::from_f64(2.0).trunc().as_f64(), 2.0);
    assert_eq!(Float::from_f64(-2.4).trunc().as_f64(), -2.0);
    assert_eq!(Float::from_f64(1999999.).trunc().as_f64(), 1999999.);
    assert_eq!(
        Float::from_f64(large_integer).trunc().as_f64(),
        large_integer
    );
    assert_eq!(Float::from_f64(0.001).trunc().as_f64(), 0.);

    // Test random values.
    let mut lfsr = Lfsr::new();
    for _ in 0..5000 {
        let v0 = f64::from_bits(lfsr.get64());
        let t0 = Float::from_f64(v0).trunc().as_f64();
        let t1 = v0.trunc();
        assert_eq!(t0.is_nan(), t1.is_nan());
        if !t1.is_nan() {
            assert_eq!(t0, t1);
        }
    }

    // Test special values.
    for val in utils::get_special_test_values() {
        let t0 = Float::from_f64(val).trunc().as_f64();
        let t1 = val.trunc();
        assert_eq!(t0.is_nan(), t1.is_nan());
        if !t1.is_nan() {
            assert_eq!(t0, t1);
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn test_round() {
    use super::utils::Lfsr;
    assert_eq!(Float::from_f64(2.0).round().as_f64(), 2.0);
    assert_eq!(Float::from_f64(2.5).round().as_f64(), 3.0);
    assert_eq!(Float::from_f64(-2.5).round().as_f64(), -3.0);

    let big_num = (1u64 << 52) as f64;
    assert_eq!(Float::from_f64(0.4).round().as_f64(), 0.);
    assert_eq!(Float::from_f64(1.4).round().as_f64(), 1.);
    assert_eq!(Float::from_f64(1.99).round().as_f64(), 2.);
    assert_eq!(Float::from_f64(2.0).round().as_f64(), 2.0);
    assert_eq!(Float::from_f64(2.1).round().as_f64(), 2.0);
    assert_eq!(Float::from_f64(-2.4).round().as_f64(), -2.0);
    assert_eq!(Float::from_f64(1999999.).round().as_f64(), 1999999.);
    assert_eq!(Float::from_f64(big_num).round().as_f64(), big_num);
    assert_eq!(Float::from_f64(0.001).round().as_f64(), 0.);

    // Test random values.
    let mut lfsr = Lfsr::new();
    for _ in 0..5000 {
        let v0 = f64::from_bits(lfsr.get64());
        let t0 = Float::from_f64(v0).round().as_f64();
        let t1 = v0.round();
        assert_eq!(t0.is_nan(), t1.is_nan());
        if !t1.is_nan() {
            assert_eq!(t0, t1);
        }
    }

    // Test special values.
    for val in utils::get_special_test_values() {
        let t0 = Float::from_f64(val).round().as_f64();
        let t1 = val.round();
        assert_eq!(t0.is_nan(), t1.is_nan());
        if !t1.is_nan() {
            assert_eq!(t0, t1);
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn test_cast_sizes() {
    use crate::FP16;
    use crate::FP256;
    let e = std::f64::consts::E;
    {
        let wide = Float::from_f64(e).cast(FP256);
        let narrow = wide.cast(FP64);
        assert_eq!(narrow.as_f64(), e);
    }

    {
        let narrow = Float::from_f64(e);
        let wide = narrow.cast(FP256);
        assert_eq!(wide.as_f64(), e);
    }

    {
        let wide = Float::from_u64(FP256, 1 << 50);
        let narrow = wide.cast(FP16);
        assert!(narrow.is_inf());
    }

    {
        let narrow = Float::from_u64(FP16, 1 << 50);
        let wide = narrow.cast(FP256);
        assert!(wide.is_inf());
    }

    {
        let narrow = Float::from_u64(FP16, 50);
        let wide = narrow.cast(FP256);
        assert_eq!(wide.as_f64(), narrow.as_f64());
        assert_eq!(wide.to_i64(), 50);
    }
}
