#[derive(Debug, Clone, Copy)]
pub struct BigInt<const PARTS: usize> {
    parts: [u64; PARTS],
}

impl<const PARTS: usize> BigInt<PARTS> {
    /// Create a new normal floating point number.
    pub fn new() -> Self {
        BigInt { parts: [0; PARTS] }
    }

    pub fn from_u64(val: u64) -> Self {
        let mut bi = BigInt { parts: [0; PARTS] };
        bi.parts[0] = val;
        bi
    }

    pub fn from_u128(val: u128) -> Self {
        let mut bi = BigInt { parts: [0; PARTS] };
        bi.parts[0] = val as u64;
        bi.parts[1] = (val >> 64) as u64;
        bi
    }

    pub fn to_u64(&self) -> u64 {
        for i in 1..PARTS {
            assert_eq!(self.parts[i], 0);
        }
        self.parts[0]
    }

    pub fn to_u128(&self) -> u128 {
        for i in 2..PARTS {
            assert_eq!(self.parts[i], 0);
        }
        (self.parts[0] as u128) + ((self.parts[1] as u128) << 64)
    }

    pub fn trunc<const P: usize>(&self) -> BigInt<P> {
        let mut n = BigInt::<P>::new();
        assert!(P <= PARTS, "Can't truncate to a larger size");
        for i in 0..PARTS {
            n.parts[i] = self.parts[i];
        }
        n
    }

    /// \returns the index of the most significant bit (the highest '1'),
    /// using 1-based counting (the first bit is 1, and zero means no bits are
    /// set).
    pub fn msb_index(&self) -> usize {
        for i in (0..PARTS).rev() {
            let part = self.parts[i];
            if part != 0 {
                let idx = 64 - part.leading_zeros() as usize;
                return i * 64 + idx;
            }
        }
        0
    }

    pub fn from_parts(parts: &[u64; PARTS]) -> Self {
        BigInt { parts: *parts }
    }

    // Add \p rhs to self, and return true if the operation overflowed.
    pub fn add(&mut self, rhs: Self) -> bool {
        let mut carry: bool = false;
        for i in 0..PARTS {
            let first = self.parts[i].overflowing_add(rhs.parts[i]);
            let second = first.0.overflowing_add(carry as u64);
            carry = first.1 || second.1;
            self.parts[i] = second.0;
        }
        carry
    }

    // Add \p rhs to self, and return true if the operation overflowed (borrow).
    pub fn sub(&mut self, rhs: Self) -> bool {
        let mut borrow: bool = false;
        for i in 0..PARTS {
            let first = self.parts[i].overflowing_sub(rhs.parts[i]);
            let second = first.0.overflowing_sub(borrow as u64);
            borrow = first.1 || second.1;
            self.parts[i] = second.0;
        }
        borrow
    }

    // multiply \p rhs to self, and return true if the operation overflowed.
    // The generic parameter \p P2 is here to work around a limitation in the
    // rust generic system. P2 needs to be set to PARTS*2.
    pub fn mul<const P2: usize>(&mut self, rhs: Self) -> bool {
        assert_eq!(P2, PARTS * 2);
        let mut parts: [u64; P2] = [0; P2];
        let mut carries: [u64; P2] = [0; P2];

        for i in 0..PARTS {
            for j in 0..PARTS {
                let pi = self.parts[i] as u128;
                let pij = pi * rhs.parts[j] as u128;

                let add0 = parts[i + j].overflowing_add(pij as u64);
                parts[i + j] = add0.0;
                carries[i + j] += add0.1 as u64;
                let add1 = parts[i + j + 1].overflowing_add((pij >> 64) as u64);
                parts[i + j + 1] = add1.0;
                carries[i + j + 1] += add1.1 as u64;
            }
        }

        let mut carry: u64 = 0;
        for i in 0..PARTS {
            let add0 = parts[i].overflowing_add(carry);
            self.parts[i] = add0.0;
            carry = add0.1 as u64 + carries[i];
        }
        for i in PARTS..P2 {
            carry |= carries[i] | parts[i];
        }

        carry > 0
    }

    // Shift the bits in the numbers \p bits to the left.
    pub fn shift_left(&mut self, bits: usize) {
        let words_to_shift = bits / u64::BITS as usize;
        let bits_in_word = bits % u64::BITS as usize;

        // If we only need to move blocks.
        if bits_in_word == 0 {
            for i in (0..PARTS).rev() {
                self.parts[i] = if i >= words_to_shift {
                    self.parts[i - words_to_shift]
                } else {
                    0
                };
            }
            return;
        }

        for i in (0..PARTS).rev() {
            let left_val = if i >= words_to_shift {
                self.parts[i - words_to_shift]
            } else {
                0
            };
            let right_val = if i > words_to_shift {
                self.parts[i - words_to_shift - 1]
            } else {
                0
            };
            let right = right_val >> (u64::BITS as usize - bits_in_word);
            let left = left_val << bits_in_word;
            self.parts[i] = left | right;
        }
    }

    // Shift the bits in the numbers \p bits to the right.
    pub fn shift_right(&mut self, bits: usize) {
        let words_to_shift = bits / u64::BITS as usize;
        let bits_in_word = bits % u64::BITS as usize;

        // If we only need to move blocks.
        if bits_in_word == 0 {
            for i in 0..PARTS {
                self.parts[i] = if i + words_to_shift < PARTS {
                    self.parts[i + words_to_shift]
                } else {
                    0
                };
            }
            return;
        }

        for i in 0..PARTS {
            let left_val = if i + words_to_shift < PARTS {
                self.parts[i + words_to_shift]
            } else {
                0
            };
            let right_val = if i + 1 + words_to_shift < PARTS {
                self.parts[i + 1 + words_to_shift]
            } else {
                0
            };
            let right = right_val << (u64::BITS as usize - bits_in_word);
            let left = left_val >> bits_in_word;
            self.parts[i] = left | right;
        }
    }

    pub fn get_part(&self, idx: usize) -> u64 {
        self.parts[idx]
    }

    pub fn dump(&self) {
        print!("[");
        for i in (0..PARTS).rev() {
            let width = u64::BITS as usize;
            print!("|{:0width$x}", self.parts[i]);
        }
        println!("]");
    }
}

impl<const PARTS: usize> Default for BigInt<PARTS> {
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn test_shl() {
    let mut x = BigInt::<4>::from_u64(0xff00ff);
    assert_eq!(x.get_part(0), 0xff00ff);
    x.shift_left(17);
    assert_eq!(x.get_part(0), 0x1fe01fe0000);
    x.shift_left(17);
    assert_eq!(x.get_part(0), 0x3fc03fc00000000);
    x.shift_left(64);
    assert_eq!(x.get_part(1), 0x3fc03fc00000000);
}

#[test]
fn test_shr() {
    let mut x = BigInt::<4>::from_u64(0xff00ff);
    x.shift_left(128);
    assert_eq!(x.get_part(2), 0xff00ff);
    x.shift_right(17);
    x.dump();
    assert_eq!(x.get_part(1), 0x807f800000000000);
    x.shift_right(17);
    x.dump();
    assert_eq!(x.get_part(1), 0x03fc03fc0000000);
    x.shift_right(64);
    x.dump();
    assert_eq!(x.get_part(0), 0x03fc03fc0000000);
}

#[test]
fn test_add_basic() {
    let mut x = BigInt::<2>::from_u64(0xffffffff00000000);
    let y = BigInt::<2>::from_u64(0xffffffff);
    let z = BigInt::<2>::from_u64(0xf);
    x.dump();
    y.dump();
    let c1 = x.add(y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    x.dump();
    let c2 = x.add(z);
    assert!(!c2);
    assert_eq!(x.get_part(0), 0xe);
    assert_eq!(x.get_part(1), 0x1);
    x.dump();
}

#[allow(dead_code)]
fn test_with_random_values(
    correct: fn(u128, u128) -> (u128, bool),
    test: fn(u128, u128) -> (u128, bool),
) {
    use super::utils::Lfsr;

    let mut lfsr = Lfsr::new();

    for _ in 0..500 {
        let v0 = lfsr.get64();
        let v1 = lfsr.get64();
        let v2 = lfsr.get64();
        let v3 = lfsr.get64();

        let n1 = (v0 as u128) + ((v1 as u128) << 64);
        let n2 = (v2 as u128) + ((v3 as u128) << 64);

        let v1 = correct(n1, n2);
        let v2 = test(n1, n2);
        assert_eq!(v1.0, v2.0, "Incorrect value");
        assert_eq!(v1.0, v2.0, "Incorrect carry");
    }
}

#[test]
fn test_sub_basic() {
    let mut x = BigInt::<2>::from_parts(&[0x0, 0x1]);
    let y = BigInt::<2>::from_u64(0x1);
    let c1 = x.sub(y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    assert_eq!(x.get_part(1), 0);
}

#[test]
fn test_basic_operations() {
    fn correct_sub(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_sub(b)
    }
    fn correct_add(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_add(b)
    }
    fn correct_mul(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_mul(b)
    }
    fn test_sub(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        let c = a.sub(b);
        (a.to_u128(), c)
    }
    fn test_add(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        let c = a.add(b);
        (a.to_u128(), c)
    }
    fn test_mul(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        let c = a.mul::<4>(b);
        (a.to_u128(), c)
    }

    test_with_random_values(correct_mul, test_mul);
    test_with_random_values(correct_add, test_add);
    test_with_random_values(correct_sub, test_sub);
}

#[test]
fn test_msb() {
    let x = BigInt::<5>::from_u64(0xffffffff00000000);
    assert_eq!(x.msb_index(), 64);

    let x = BigInt::<5>::from_u64(0x0);
    assert_eq!(x.msb_index(), 0);

    let x = BigInt::<5>::from_u64(0x1);
    assert_eq!(x.msb_index(), 1);

    let mut x = BigInt::<5>::from_u64(0x1);
    x.shift_left(189);
    assert_eq!(x.msb_index(), 189 + 1);

    for i in 0..256 {
        let mut x = BigInt::<5>::from_u64(0x1);
        x.shift_left(i);
        assert_eq!(x.msb_index(), i + 1);
    }
}
