/// \returns a mask full of 1s, of \p b bits.
pub fn mask(b: usize) -> usize {
    (1 << (b)) - 1
}

#[test]
fn test_masking() {
    assert_eq!(mask(0), 0x0);
    assert_eq!(mask(1), 0x1);
    assert_eq!(mask(8), 255);
}

/// Convert a mantissa in the implicit format (no possible leading 1 bit) to
/// the internal storage format. If \p leading_1 is set then a leading one is
/// added (otherwise it is a subnormal).
/// Format: [1 IIIIII 00000000]
pub fn expand_mantissa_to_explicit<const FROM: usize>(
    input: u64,
    leading_1: bool,
) -> u64 {
    let value: u64 = if leading_1 { 1 << 63 } else { 0 };
    let shift = 63 - FROM;
    value | (input << shift)
}

#[test]
fn test_expand_mantissa() {
    assert_eq!(expand_mantissa_to_explicit::<8>(0, true), 1 << 63);
    assert_eq!(
        expand_mantissa_to_explicit::<8>(1, true),
        0x8080000000000000
    );
    assert_eq!(
        expand_mantissa_to_explicit::<32>(0xffffffff, false),
        0x7fffffff80000000
    );
}

#[derive(Debug, Clone, Copy)]
pub enum RoundMode {
    Trunc,
    Even,
}

// \returns the bias for this Float type.
pub fn compute_ieee745_bias(exponent_bits: usize) -> usize {
    (1 << (exponent_bits - 1)) - 1
}

/// \returns list of interesting values that various tests use to catch edge cases.
pub fn get_special_test_values() -> [f64; 20] {
    [
        -f64::NAN,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::EPSILON,
        -f64::EPSILON,
        0.000000000000000000000000000000000000001,
        f64::MIN,
        f64::MAX,
        std::f64::consts::PI,
        std::f64::consts::LN_2,
        std::f64::consts::SQRT_2,
        std::f64::consts::E,
        0.0,
        -0.0,
        10.,
        -10.,
        -0.00001,
        0.1,
        355. / 113.,
    ]
}

// Linear-feedback shift register.
pub struct Lfsr {
    state: u32,
}

impl Default for Lfsr {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfsr {
    pub fn new() -> Lfsr {
        Lfsr { state: 0x13371337 }
    }

    pub fn next(&mut self) {
        let a = (self.state >> 24) & 1;
        let b = (self.state >> 23) & 1;
        let c = (self.state >> 22) & 1;
        let d = (self.state >> 17) & 1;
        let n = a ^ b ^ c ^ d ^ 1;
        self.state <<= 1;
        self.state |= n;
    }

    pub fn get(&mut self) -> u32 {
        let mut res: u32 = 0;
        for _ in 0..32 {
            self.next();
            res <<= 1;
            res ^= self.state & 0x1;
        }
        res
    }

    pub fn get64(&mut self) -> u64 {
        ((self.get() as u64) << 32) | self.get() as u64
    }
}

#[test]
fn test_lfsr_balance() {
    let mut lfsr = Lfsr::new();

    // Count the number of items, and the number of 1s.
    let mut items = 0;
    let mut ones = 0;

    for _ in 0..10000 {
        let mut u = lfsr.get();
        for _ in 0..32 {
            items += 1;
            ones += u & 1;
            u >>= 1;
        }
    }
    // Make sure that we have around 50% 1s and 50% zeros.
    assert!((ones as f64) < (0.55 * items as f64));
    assert!((ones as f64) > (0.45 * items as f64));
}
#[test]
fn test_repetition() {
    let mut lfsr = Lfsr::new();
    let first = lfsr.get();
    let second = lfsr.get();

    // Make sure that the items don't repeat themselves too frequently.
    for _ in 0..30000 {
        assert_ne!(first, lfsr.get());
        assert_ne!(second, lfsr.get());
    }
}
