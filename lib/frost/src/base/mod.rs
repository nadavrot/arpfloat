/// \returns a mask full of 1s, of \p b bits.
fn mask(b: usize) -> usize {
    (1 << (b)) - 1
}

#[test]
fn test_masking() {
    assert_eq!(mask(0), 0x0);
    assert_eq!(mask(1), 0x1);
    assert_eq!(mask(8), 255);
}

/// Add the hidden bit to the significant.
fn add_msb_bit(num: u64, bits: u64) -> u64 {
    assert!(num < (1 << bits));
    (1 << bits) + num
}

#[test]
fn test_expand() {
    assert_eq!(add_msb_bit(127, 7), 255);
    assert_eq!(add_msb_bit(0, 1), 2);
    assert_eq!(add_msb_bit(15, 10), 1024 + 15);
}

/// Cast \p input from \p from to \p to bits, and preserve the MSB bits.
fn cast_msb_values<const FROM: usize, const TO: usize>(input: u64) -> u64 {
    if FROM > TO {
        // [....xxxxxxxx]
        // [.........yyy]
        return input >> (FROM - TO);
    }
    // [.......xxxxx]
    // [....yyyyyyyy]
    input << (TO - FROM)
}

#[test]
fn test_msb_cast() {
    assert_eq!(cast_msb_values::<8, 32>(0xff), 0xff000000);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Float<const EXPONENT: usize, const SIGNIFICANT: usize> {
    // The Sign bit.
    sign: bool,
    // The Exponent.
    exp: u64,
    // The Integral Significant.
    sig: u64,
}

impl<const EXPONENT: usize, const SIGNIFICANT: usize>
    Float<EXPONENT, SIGNIFICANT>
{
    pub fn new(sign: bool, exp: i64, sig: u64) -> Float<EXPONENT, SIGNIFICANT> {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_exp(exp);
        a.set_significant(sig);
        a
    }

    pub fn inf(sign: bool) -> Float<EXPONENT, SIGNIFICANT> {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_unbiased_exp(mask(EXPONENT) as u64);
        a.set_significant(0);
        a
    }
    pub fn nan(sign: bool) -> Float<EXPONENT, SIGNIFICANT> {
        let mut a = Self::default();
        a.set_sign(sign);
        a.set_unbiased_exp(mask(EXPONENT) as u64);
        a.set_significant((1 << SIGNIFICANT) - 1);
        a
    }

    /// \returns True if the Float has the signaling exponent.
    fn in_special_exp(&self) -> bool {
        self.get_unbiased_exp() == mask(EXPONENT) as u64
    }

    /// \returns True if the Float is negative
    pub fn is_negative(&self) -> bool {
        self.get_sign()
    }

    /// \returns True if the Float is a positive or negative infinity.
    pub fn is_inf(&self) -> bool {
        self.in_special_exp() && self.get_significant() == 0
    }

    /// \returns True if the Float is a positive or negative NaN.
    pub fn is_nan(&self) -> bool {
        self.in_special_exp() && self.get_significant() != 0
    }

    pub fn from_f32(float: f32) -> Float<EXPONENT, SIGNIFICANT> {
        Self::from_bits::<8, 23>(float.to_bits() as u64)
    }

    pub fn from_f64(float: f64) -> Float<EXPONENT, SIGNIFICANT> {
        Self::from_bits::<11, 52>(float.to_bits())
    }

    pub fn from_bits<const E: usize, const S: usize>(
        float: u64,
    ) -> Float<EXPONENT, SIGNIFICANT> {
        // Extract the biased exponent (wipe the sign and significant).
        let biased_exp = (float >> S) & mask(E) as u64;
        // Wipe the exponent and significant.
        let sign = (float >> (E + S)) & 1;
        // Wipe the sign and exponent.
        let integral = float & mask(S) as u64;

        let mut a = Self::default();
        a.set_sign(sign == 1);
        a.set_exp(biased_exp as i64 - Self::get_float_bias(E) as i64);
        let integral = cast_msb_values::<S, SIGNIFICANT>(integral);
        a.set_significant(integral);
        a
    }

    /// \returns the sign bit.
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    /// Sets the sign to \p s.
    pub fn set_sign(&mut self, s: bool) {
        self.sign = s;
    }

    /// \returns the significant (without the leading 1).
    pub fn get_significant(&self) -> u64 {
        let max_sig: u64 = 1 << SIGNIFICANT;
        assert!(self.sig <= max_sig, "Significant out of range");
        self.sig
    }

    /// Sets the significant to \p sg.
    pub fn set_significant(&mut self, sg: u64) {
        let max_sig: u64 = 1 << SIGNIFICANT;
        assert!(sg <= max_sig, "Significant out of range");
        self.sig = sg;
    }

    /// \returns the unbiased exponent.
    pub fn get_unbiased_exp(&self) -> u64 {
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
    pub fn set_unbiased_exp(&mut self, new_exp: u64) {
        self.exp = new_exp
    }
    // \returns the bias for this Float type.
    pub fn get_float_bias(exponent_bits: usize) -> usize {
        (1 << (exponent_bits - 1)) - 1
    }

    pub fn get_bias() -> u64 {
        Self::get_float_bias(EXPONENT) as u64
    }

    /// \returns the bounds of the upper and lower bounds of the exponent.
    pub fn get_exp_bounds() -> (i64, i64) {
        let exp_min: i64 = -(Self::get_bias() as i64);
        let exp_max: i64 = ((1 << EXPONENT) - Self::get_bias()) as i64;
        (exp_min, exp_max)
    }

    pub fn cast<const E: usize, const S: usize>(&self) -> Float<E, S> {
        let mut x = Float::<E, S>::default();
        x.set_sign(self.get_sign());
        x.set_exp(self.get_exp());
        // Handle Nan/Inf.
        if self.in_special_exp() {
            x.set_unbiased_exp(mask(E) as u64);
        }
        let sig = cast_msb_values::<SIGNIFICANT, S>(self.get_significant());
        x.set_significant(sig);
        x
    }

    fn as_native_float<const E: usize, const S: usize>(&self) -> u64 {
        assert!(SIGNIFICANT == 23);
        assert!(EXPONENT == 8);
        // https://en.wikipedia.org/wiki/IEEE_754
        let mut bits: u64 = self.get_sign() as u64;
        bits <<= E;
        bits |= (self.get_exp() + Self::get_bias() as i64) as u64;
        bits <<= S;
        let sig = self.get_significant();
        assert!(sig <= 1 << S);
        bits |= sig;
        bits
    }
    pub fn as_f32(&self) -> f32 {
        let b: FP32 = self.cast();
        let bits = b.as_native_float::<8, 23>();
        f32::from_bits(bits as u32)
    }
    pub fn as_f64(&self) -> f64 {
        let b: FP64 = self.cast();
        let bits = b.as_native_float::<11, 52>();
        f64::from_bits(bits)
    }
    pub fn dump(&self) {
        let exp = self.get_exp();
        let significant = self.get_significant();
        let sign = self.get_sign() as usize;
        println!(
            "FP[S={} : E={} (biased {}) :SI=0x{:x}]",
            sign, self.exp, exp, significant
        );
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

#[test]
fn test_conversion_wide_range() {
    for i in 0..(1 << 16) {
        let val = f32::from_bits(i << 16);
        let a = FP64::from_f32(val);
        let b: FP32 = a.cast();
        let res = b.as_f32();
        assert_eq!(res.to_bits(), (i << 16));
    }
}

#[test]
fn constructor_test() {
    let values: [u32; 5] =
        [0x3f8fffff, 0x40800000, 0x3f000000, 0xc60b40ec, 0xbc675793];

    for i in 0..5 {
        let output = f32::from_bits(values[i]);
        let a = FP64::from_f32(output);
        let b: FP32 = a.cast();
        assert_eq!(a.as_f32(), output);
        assert_eq!(b.as_f32(), output);
    }
}

#[test]
fn test_nan_inf() {
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
        let mut a = FP64::from_f64(f64::from_bits((mask(32) << 32) as u64));
        assert!(!a.is_inf());
        assert!(a.is_nan());
        a.set_significant(0);
        assert!(a.is_inf());
        assert!(!a.is_nan());
        assert!(a.is_negative());
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

// See Chapter 8. Algorithms for the Five Basic Operations -- Pg 248
pub fn add<const E: usize, const S: usize>(
    x: Float<E, S>,
    y: Float<E, S>,
) -> Float<E, S> {
    assert!(x.get_exp() >= y.get_exp());
    assert!(!x.in_special_exp() && !y.in_special_exp());

    // Significant alignment.
    let exp_delta = x.get_exp() - y.get_exp();
    let mut er = x.get_exp();
    // Addition of the significant.

    let y_significant =
        add_msb_bit(y.get_significant(), S as u64) >> (exp_delta);
    let x_significant = add_msb_bit(x.get_significant(), S as u64);

    let mut is_neg = x.is_negative();

    let is_plus = x.get_sign() == y.get_sign();
    let mut xy_significant = if is_plus {
        x_significant + y_significant
    } else if y_significant > x_significant {
        is_neg ^= true;
        y_significant - x_significant
    } else {
        x_significant - y_significant
    };

    let overflow = xy_significant >> S;

    // Handle the case where there was a carry out in the significant addition.
    match overflow {
        0 => {
            // Cancellation happened, we need to normalize the number.
            // Shift xy_significant to the left, and subtract from the exponent
            // until you underflow or until xy_sig is normalized.
            let lz = xy_significant.leading_zeros() as u64;
            let expected_zeros = 64 - (S + 1) as u64;
            let lz = lz - expected_zeros;

            let lower_bound = Float::<E, S>::get_exp_bounds().0;
            // How far can we lower the exponent.
            let delta_to_min = er - lower_bound;

            let shift = delta_to_min.min(lz as i64);
            xy_significant <<= shift;
            er -= shift;
        }
        1 => {
            // Nothing to do.
        }
        2 => {
            xy_significant >>= 1;
            er += 1;
        }
        _default => {
            panic!("Invalid overflow value");
        }
    }

    // TODO: handle the case where there was a cancellation in the significant
    // addition.

    let mut r = Float::<E, S>::default();
    r.set_significant(xy_significant & mask(S) as u64);
    r.set_exp(er);
    r.set_sign(is_neg);
    r
}

#[test]
fn test_addition() {
    fn add_helper(a: f32, b: f32) -> f32 {
        let a = FP32::from_f32(a);
        let b = FP32::from_f32(b);
        let c = add(a, b);
        c.as_f32()
    }

    assert_eq!(add_helper(8., -4.), 4.);
    assert_eq!(add_helper(8., 4.), 12.);
    assert_eq!(add_helper(128., 2.), 130.);
    assert_eq!(add_helper(128., -8.), 120.);
    assert_eq!(add_helper(64., -60.), 4.);
    assert_eq!(add_helper(69., -65.), 4.);
    assert_eq!(add_helper(69., 69.), 138.);
    assert_eq!(add_helper(69., 1.), 70.);
    assert_eq!(add_helper(-128., -8.), -136.);
    assert_eq!(add_helper(64., -65.), -1.);
}
