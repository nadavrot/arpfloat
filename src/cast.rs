use super::bigint::LossFraction;
use super::float::{self, Category};
use super::float::{Float, MantissaTy, RoundingMode, FP32, FP64};
use super::utils;
use super::utils::mask;

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Load the integer `val` into the float. Notice that the number may
    /// overflow, or rounded to the nearest even integer.
    pub fn from_u64(val: u64) -> Self {
        let val = MantissaTy::from_u64(val);
        let mut a = Self::new(false, MANTISSA as i64, val);
        a.normalize(RoundingMode::NearestTiesToEven, LossFraction::ExactlyZero);
        a
    }

    /// Load the integer `val` into the float. Notice that the number may
    /// overflow, or rounded to the nearest even integer.
    pub fn from_i64(val: i64) -> Self {
        if val < 0 {
            let mut a = Self::from_u64(-val as u64);
            a.set_sign(true);
            return a;
        }

        Self::from_u64(val as u64)
    }

    /// Converts and returns the rounded integral part.
    pub fn to_i64(&self, rm: RoundingMode) -> i64 {
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
            return *self;
        }

        let exp = self.get_exp();

        if exp > MANTISSA as i64 {
            // Already an integer.
            return *self;
        }

        // Numbers that are smaller than 1 are rounded to zero.
        if exp < -1 {
            return Self::zero(self.get_sign());
        }

        // This is a fraction. Figure out which bits represent values over one
        // and clear out the values that represent the fraction.
        let trim = (MANTISSA as i64 - exp) as usize;
        let mut m = self.get_mantissa();
        m.shift_right(trim);
        m.shift_left(trim);
        Self::new(self.get_sign(), self.get_exp(), m)
    }

    fn convert_normal_to_integer(&self, rm: RoundingMode) -> MantissaTy {
        // We are converting to integer, so set the center point of the exponent
        // to the lsb instead of the msb.
        let i_exp = self.get_exp() - MANTISSA as i64;
        if i_exp < 0 {
            let (mut m, loss) = float::shift_right_with_loss(
                self.get_mantissa(),
                -i_exp as u64,
            );

            if self.need_round_away_from_zero(rm, loss) {
                let _ = m.inplace_add(&MantissaTy::one());
            }
            m
        } else {
            let mut m = self.get_mantissa();
            m.shift_left(i_exp as usize);
            m
        }
    }

    fn from_bits(float: u64) -> Self {
        // Extract the biased exponent (wipe the sign and mantissa).
        let biased_exp = ((float >> MANTISSA) & mask(EXPONENT) as u64) as i64;
        // Wipe the original exponent and mantissa.
        let sign = (float >> (EXPONENT + MANTISSA)) & 1;
        // Wipe the sign and exponent.
        let mut mantissa = float & mask(MANTISSA) as u64;

        let sign = sign == 1;

        // Check for NaN/Inf
        if biased_exp == mask(EXPONENT) as i64 {
            if mantissa == 0 {
                return Self::inf(sign);
            }
            return Self::nan(sign);
        }

        let mut exp = biased_exp - Self::get_bias();

        // Add the implicit bit for normal numbers.
        if biased_exp != 0 {
            mantissa += 1u64 << MANTISSA;
        } else {
            // Handle denormals, adjust the exponent to the legal range.
            exp += 1;
        }

        let mantissa = MantissaTy::from_u64(mantissa);
        Self::new(sign, exp, mantissa)
    }

    /// Cast to another float using the rounding mode `rm`.
    pub fn cast_with_rm<const E: usize, const M: usize>(
        &self,
        rm: RoundingMode,
    ) -> Float<E, M> {
        let exp_delta = MANTISSA as i64 - M as i64;
        let mut x = Float::<E, M>::raw(
            self.get_sign(),
            self.get_exp() - exp_delta,
            self.get_mantissa(),
            self.get_category(),
        );
        // Don't normalize if this is a nop conversion.
        if E != EXPONENT && M != MANTISSA {
            x.normalize(rm, LossFraction::ExactlyZero);
        }
        x
    }
    /// Convert from one float format to another.
    pub fn cast<const E: usize, const M: usize>(&self) -> Float<E, M> {
        self.cast_with_rm(RoundingMode::NearestTiesToEven)
    }

    fn as_native_float(&self) -> u64 {
        // https://en.wikipedia.org/wiki/IEEE_754
        let mantissa: u64;
        let mut exp: u64;
        match self.get_category() {
            Category::Infinity => {
                mantissa = 0;
                exp = mask(EXPONENT) as u64;
            }
            Category::NaN => {
                mantissa = 1 << (MANTISSA - 1);
                exp = mask(EXPONENT) as u64;
            }
            Category::Zero => {
                mantissa = 0;
                exp = 0;
            }
            Category::Normal => {
                exp = (self.get_exp() + Self::get_bias()) as u64;
                debug_assert!(exp > 0);
                let m = self.get_mantissa().as_u64();
                // Encode denormals. If the exponent is the minimum value and we
                // don't have a leading integer bit (in the form 1.mmmm) then
                // this is a denormal value and we need to encode it as such.
                if (exp == 1) && ((m >> MANTISSA) == 0) {
                    exp = 0;
                }
                mantissa = m & utils::mask(MANTISSA) as u64;
            }
        }

        let mut bits: u64 = self.get_sign() as u64;
        bits <<= EXPONENT;
        bits |= exp;
        bits <<= MANTISSA;
        debug_assert!(mantissa <= 1 << MANTISSA);
        bits |= mantissa;
        bits
    }
    // Convert this float to fp32. Notice that the number may overflow or
    // rounded to the nearest even (see cast and cast_with_rm).
    pub fn as_f32(&self) -> f32 {
        let b: FP32 = self.cast();
        let bits = b.as_native_float();
        f32::from_bits(bits as u32)
    }
    // Convert this float to fp64. Notice that the number may overflow or
    // rounded to the nearest even (see cast and cast_with_rm).
    pub fn as_f64(&self) -> f64 {
        let b: FP64 = self.cast();
        let bits = b.as_native_float();
        f64::from_bits(bits)
    }

    // Loads and converts a native fp32 value. Notice that the number may overflow or
    // rounded to the nearest even (see cast and cast_with_rm).
    pub fn from_f32(float: f32) -> Self {
        FP32::from_bits(float.to_bits() as u64).cast()
    }

    // Loads and converts a native fp64 value. Notice that the number may overflow or
    // rounded to the nearest even (see cast and cast_with_rm).
    pub fn from_f64(float: f64) -> Self {
        FP64::from_bits(float.to_bits()).cast()
    }
}

#[test]
fn test_rounding_to_integer() {
    // Test the low integers with round-to-zero.
    for i in 0..100 {
        let r = FP64::from_f64(i as f64 + 0.1).to_i64(RoundingMode::Zero);
        assert_eq!(i, r);
    }

    // Test the high integers with round_to_zero.
    for i in 0..100 {
        let val = (i as i64) << 54;
        let r = FP64::from_i64(val).to_i64(RoundingMode::Zero);
        assert_eq!(val, r);
    }

    use RoundingMode::NearestTiesToAway;
    assert_eq!(1, FP64::from_f64(0.5).to_i64(NearestTiesToAway));
    assert_eq!(0, FP64::from_f64(0.49).to_i64(NearestTiesToAway));
    assert_eq!(199999, FP64::from_f64(199999.49).to_i64(NearestTiesToAway));
    assert_eq!(0, FP64::from_f64(-0.49).to_i64(NearestTiesToAway));
    assert_eq!(-1, FP64::from_f64(-0.5).to_i64(NearestTiesToAway));

    use RoundingMode::Zero;
    assert_eq!(0, FP64::from_f64(0.9).to_i64(Zero));
    assert_eq!(1, FP64::from_f64(1.1).to_i64(Zero));
    assert_eq!(99, FP64::from_f64(99.999).to_i64(Zero));
    assert_eq!(0, FP64::from_f64(-0.99).to_i64(Zero));
    assert_eq!(0, FP64::from_f64(-0.5).to_i64(Zero));

    use RoundingMode::Positive;
    assert_eq!(1, FP64::from_f64(0.9).to_i64(Positive));
    assert_eq!(2, FP64::from_f64(1.1).to_i64(Positive));
    assert_eq!(100, FP64::from_f64(99.999).to_i64(Positive));
    assert_eq!(0, FP64::from_f64(-0.99).to_i64(Positive));
    assert_eq!(0, FP64::from_f64(-0.5).to_i64(Positive));

    // Special values
    use RoundingMode::NearestTiesToEven;
    let n_inf = f64::NEG_INFINITY;
    let inf = f64::INFINITY;
    assert_eq!(0, FP64::from_f64(f64::NAN).to_i64(NearestTiesToEven));
    assert_eq!(i64::MIN, FP64::from_f64(n_inf).to_i64(NearestTiesToEven));
    assert_eq!(i64::MAX, FP64::from_f64(inf).to_i64(NearestTiesToEven));
}

#[test]
fn test_round_trip_native_float_cast() {
    let f = f32::from_bits(0x41700000);
    let a = FP32::from_f32(f);
    assert_eq!(f, a.as_f32());

    let pi = 355. / 113.;
    let a = FP64::from_f64(pi);
    assert_eq!(pi, a.as_f64());

    assert!(FP64::from_f64(f64::NAN).is_nan());
    assert!(!FP64::from_f64(f64::NAN).is_inf());
    assert!(FP64::from_f64(f64::INFINITY).is_inf());
    assert!(!FP64::from_f64(f64::INFINITY).is_nan());
    assert!(FP64::from_f64(f64::NEG_INFINITY).is_inf());

    let a_float = f32::from_bits(0x3f8fffff);
    let a = FP64::from_f32(a_float);
    let b: FP32 = a.cast();
    assert_eq!(a.as_f32(), a_float);
    assert_eq!(b.as_f32(), a_float);

    let f = f32::from_bits(0x000000);
    let a = FP32::from_f32(f);
    assert!(!a.is_normal());
    assert_eq!(f, a.as_f32());
}

#[test]
fn test_cast_easy_ctor() {
    let values = [0x3f8fffff, 0x40800000, 0x3f000000, 0xc60b40ec, 0xbc675793];

    for v in values {
        let output = f32::from_bits(v);
        let a = FP64::from_f32(output);
        let b: FP32 = a.cast();
        assert_eq!(a.as_f32(), output);
        assert_eq!(b.as_f32(), output);
    }
}

#[test]
fn test_cast_from_integers() {
    use super::float::FP16;

    let pi = 355. / 133.;
    let e = 193. / 71.;

    assert_eq!(FP32::from_i64(1 << 32).as_f32(), (1u64 << 32) as f32);
    assert_eq!(FP32::from_i64(1 << 34).as_f32(), (1u64 << 34) as f32);
    assert_eq!(FP32::from_f64(pi).as_f32(), (pi) as f32);
    assert_eq!(FP32::from_f64(e).as_f32(), (e) as f32);
    assert_eq!(FP32::from_u64(8388610).as_f32(), 8388610 as f32);

    for i in 0..(1 << 16) {
        assert_eq!(FP32::from_u64(i << 12).as_f32(), (i << 12) as f32);
    }

    assert_eq!(FP64::from_i64(0).as_f64(), 0.);
    assert_eq!(FP16::from_i64(65500).as_f64(), 65504.0);
    assert_eq!(FP16::from_i64(65504).as_f64(), 65504.0);
    assert_eq!(FP16::from_i64(65519).as_f64(), 65504.0);
    assert_eq!(FP16::from_i64(65520).as_f64(), f64::INFINITY);
    assert_eq!(FP16::from_i64(65536).as_f64(), f64::INFINITY);

    for i in -100..100 {
        let a = FP32::from_i64(i);
        let b = FP32::from_f64(i as f64);
        assert_eq!(a.as_f32(), b.as_f32());
    }
}

#[test]
fn test_cast_zero_nan_inf() {
    assert!(FP64::nan(true).as_f64().is_nan());
    assert_eq!(FP64::zero(false).as_f64(), 0.0);
    assert_eq!(FP64::zero(true).as_f64(), -0.0);

    assert!(FP64::nan(true).is_nan());
    assert!(FP64::inf(true).is_inf());
    {
        let a = FP32::from_f32(f32::from_bits(0x3f8fffff));
        assert!(!a.is_inf());
        assert!(!a.is_nan());
        assert!(!a.is_negative());
    }
    {
        let a = FP32::from_f32(f32::from_bits(0xf48fffff));
        assert!(!a.is_inf());
        assert!(!a.is_nan());
        assert!(a.is_negative());
    }
    {
        let a = FP32::from_f32(f32::from_bits(0xff800000)); // -Inf
        assert!(a.is_inf());
        assert!(!a.is_nan());
        assert!(a.is_negative());
    }
    {
        let a = FP32::from_f32(f32::from_bits(0xffc00000)); // -Nan.
        assert!(!a.is_inf());
        assert!(a.is_nan());
        assert!(a.is_negative());
    }

    {
        let a = FP64::from_f64(f64::from_bits((mask(32) << 32) as u64));
        assert!(!a.is_inf());
        assert!(a.is_nan());
    }
    {
        // Check that casting propagates inf/nan.
        let a = FP32::from_f32(f32::from_bits(0xff800000)); // -Inf
        let b: FP64 = a.cast();
        assert!(b.is_inf());
        assert!(!b.is_nan());
        assert!(b.is_negative());
    }
}

#[test]
fn test_cast_down_easy() {
    // Check that we can cast the numbers down, matching the hardware casting.
    for v in [0.3, 0.1, 14151241515., 14151215., 0.0000000001, 1000000000.] {
        let res = FP64::from_f64(v).as_f32();
        assert_eq!(FP64::from_f64(v).as_f64().to_bits(), v.to_bits());
        assert!(res == v as f32);
    }
}

#[test]
fn test_load_store_all_f32() {
    // Try to load and store normals and denormals.
    for i in 0..(1u64 << 16) {
        let in_f = f32::from_bits((i << 10) as u32);
        let fp_f = FP32::from_f32(in_f);
        let out_f = fp_f.as_f32();
        assert_eq!(in_f.is_nan(), out_f.is_nan());
        assert_eq!(in_f.is_infinite(), out_f.is_infinite());
        assert!(in_f.is_nan() || (in_f.to_bits() == out_f.to_bits()));
    }
}

#[test]
fn test_cast_down_complex() {
    // Try casting a bunch of difficult values such as inf, nan, denormals, etc.
    for v in utils::get_special_test_values() {
        let res = FP64::from_f64(v).as_f32();
        assert_eq!(FP64::from_f64(v).as_f64().to_bits(), v.to_bits());
        assert_eq!(v.is_nan(), res.is_nan());
        assert!(v.is_nan() || res == v as f32);
    }
}

#[test]
fn test_round_floor() {
    let large_integer = (1u64 << 52) as f64;
    assert_eq!(FP64::from_f64(0.4).trunc().as_f64(), 0.);
    assert_eq!(FP64::from_f64(1.4).trunc().as_f64(), 1.);
    assert_eq!(FP64::from_f64(1.99).trunc().as_f64(), 1.);
    assert_eq!(FP64::from_f64(2.0).trunc().as_f64(), 2.0);
    assert_eq!(FP64::from_f64(-2.4).trunc().as_f64(), -2.0);
    assert_eq!(FP64::from_f64(1999999.).trunc().as_f64(), 1999999.);
    assert_eq!(
        FP64::from_f64(large_integer).trunc().as_f64(),
        large_integer
    );
    assert_eq!(FP64::from_f64(0.001).trunc().as_f64(), 0.);
}
