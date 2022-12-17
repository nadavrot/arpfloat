#[derive(Debug, Clone, Copy)]
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
    pub fn default() -> Float<EXPONENT, SIGNIFICANT> {
        Float {
            sign: false,
            exp: 0,
            sig: 0,
        }
    }

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
        let new_exp: i64 = new_exp + (Self::get_bias() as i64);
        assert!(new_exp <= exp_max);
        assert!(exp_min <= new_exp);
        self.exp = new_exp as u64
    }

    pub fn get_bias() -> u64 {
        (1 << (EXPONENT - 1)) - 1
    }

    pub fn as_f64(&self) -> f64 {
        let exp = self.get_exp();
        let significant = (1 << SIGNIFICANT) + self.get_significant();
        let shift = 1 << SIGNIFICANT;
        let significant: f64 = (significant as f64) / (shift as f64);
        let e2 = f64::powf(2., exp as f64);

        let mut val: f64 = significant * e2;
        if self.get_sign() {
            val = -val;
        }
        val
    }

    pub fn dump(&self) {
        let exp = self.get_exp();
        let significant = self.get_significant();
        let sign = self.get_sign() as usize;
        println!("FP[{} : {} : {}]", sign, exp, significant);
    }
}

pub type FP16 = Float<5, 11>;
pub type FP32 = Float<8, 24>;
pub type FP64 = Float<11, 43>;

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
    let inputs: [u32; 5] = [0x3f800000, 0x40800000, 0x3f000000, 4, 5];
    let outputs: [f64; 5] = [1.0, 4.0, 0.5, 4., 5.];

    for i in 0..3 {
        let a = FP32::from_u32_float(inputs[i]);
        a.dump();
        assert_eq!(a.as_f64(), outputs[i]);
    }
}
