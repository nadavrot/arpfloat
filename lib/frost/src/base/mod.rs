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

    pub fn from_u32_float(float: u32) -> Float<EXPONENT, SIGNIFICANT> {
        let integral = float & 0x7fffff;
        let exp: i64 = ((float >> 23) & 0xff) as i64;
        let sign = float >> 31;
        let mut a = Self::default();
        a.set_sign(sign == 1);
        a.set_exp(exp - 127);
        a.set_significant(integral as u64);
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
        self.sig
    }

    /// Sets the significant to \p sg.
    pub fn set_significant(&mut self, sg: u64) {
        let max_sig: u64 = 1 << SIGNIFICANT;
        assert!(sg < max_sig, "Significant out of range");
        self.sig = sg;
    }

    /// \returns the biased exponent.
    pub fn get_exp(&self) -> i64 {
        self.exp as i64 - Self::get_bias() as i64
    }

    /// Sets the biased exponent to \p new_exp.
    pub fn set_exp(&mut self, new_exp: i64) {
        let exp_min: i64 = -(Self::get_bias() as i64);
        let exp_max: i64 = ((1 << EXPONENT) - Self::get_bias()) as i64;
        assert!(new_exp <= exp_max);
        assert!(exp_min <= new_exp);

        let new_exp: i64 = new_exp + (Self::get_bias() as i64);
        self.exp = new_exp as u64
    }

    pub fn get_bias() -> u64 {
        (1 << (EXPONENT - 1)) - 1
    }

    /// Cast \p input from \p from to \p to bits, and preserve the MSB bits.
    fn cast_msb_values(input: u64, from: usize, to: usize) -> u64 {
        if from > to {
            // [....xxxxxxxx]
            // [.........yyy]
            return input >> (from - to);
        }
        // [.......xxxxx]
        // [....yyyyyyyy]
        input << (to - from)
    }

    pub fn cast<const E: usize, const S: usize>(&self) -> Float<E, S> {
        let mut x = Float::<E, S>::default();
        x.set_sign(self.get_sign());
        x.set_exp(self.get_exp());
        let sig = Self::cast_msb_values(self.get_significant(), SIGNIFICANT, S);
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
        assert!(sig < 1 << S);
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
            "FP[S={} : E={} (biased {}) :SI={}]",
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
fn constructor_test() {
    let values: [u32; 5] =
        [0x3f8fffff, 0x40800000, 0x3f000000, 0xc60b40ec, 0xbc675793];

    for i in 0..5 {
        let output = f32::from_bits(values[i]);
        let k = output as f64 as f32;
        assert!(k == output);
        let a = FP64::from_u32_float(values[i]);
        let b: FP32 = a.cast();
        a.dump();
        b.dump();
        println!(
            "{:x} == {:x} == {:x}",
            b.as_f32().to_bits(),
            (output).to_bits(),
            values[i]
        );
        assert_eq!(a.as_f32(), output);
    }
}
