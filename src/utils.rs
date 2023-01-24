//! This file contains simple helper functions and test helpers.

/// Returns a mask full of 1s, of `b` bits.
pub fn mask(b: usize) -> usize {
    (1 << (b)) - 1
}

#[test]
fn test_masking() {
    assert_eq!(mask(0), 0x0);
    assert_eq!(mask(1), 0x1);
    assert_eq!(mask(8), 255);
}

#[cfg(feature = "std")]
#[allow(dead_code)]
/// Returns list of interesting values that various tests use to catch edge cases.
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

// Linear-feedback shift register. We use this as a random number generator for
// tests.
pub struct Lfsr {
    state: u32,
}

impl Default for Lfsr {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfsr {
    /// Generate a new LFSR number generator.
    pub fn new() -> Lfsr {
        Lfsr { state: 0x13371337 }
    }

    /// Generate a new LFSR number generator that starts with a specific state.
    pub fn new_with_seed(seed: u32) -> Lfsr {
        Lfsr {
            state: 0x13371337 ^ seed,
        }
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

    fn get(&mut self) -> u32 {
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

// Implement `Iterator` for `Lfsr`.
impl Iterator for Lfsr {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.get64())
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

// Multiply a and b, and return the (low, high) parts.
#[allow(dead_code)]
fn mul_part(a: u64, b: u64) -> (u64, u64) {
    let half_bits = u64::BITS / 2;
    let half_mask = (1 << half_bits) - 1;

    let a_lo = a & half_mask;
    let a_hi = a >> half_bits;
    let b_lo = b & half_mask;
    let b_hi = b >> half_bits;

    let ab_hi = a_hi * b_hi;
    let ab_mid = a_hi * b_lo;
    let ba_mid = b_hi * a_lo;
    let ab_low = a_lo * b_lo;

    let carry =
        ((ab_mid & half_mask) + (ba_mid & half_mask) + (ab_low >> half_bits))
            >> half_bits;
    let low = (ab_mid << half_bits)
        .overflowing_add(ba_mid << half_bits)
        .0
        .overflowing_add(ab_low)
        .0;

    let high = (ab_hi + (ab_mid >> half_bits) + (ba_mid >> half_bits)) + carry;
    (low, high)
}

#[test]
fn test_mul_parts() {
    use super::utils::Lfsr;

    let mut lfsr = Lfsr::new();

    for _ in 0..500 {
        let v0 = lfsr.get64();
        let v1 = lfsr.get64();
        let res = mul_part(v0, v1);
        let full = v0 as u128 * v1 as u128;
        assert_eq!(full as u64, res.0);
        assert_eq!((full >> 64) as u64, res.1);
    }
}
