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

    pub fn trunc<const P: usize>(&self) -> BigInt<P> {
        let mut n = BigInt::<P>::new();
        assert!(P <= PARTS, "Can't truncate to a larger size");
        for i in 0..PARTS {
            n.parts[i] = self.parts[i];
        }
        n
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

#[test]
fn test_mul_random_vals() {
    use super::utils::Lfsr;

    let mut lfsr = Lfsr::new();

    for _ in 0..500 {
        let v0 = lfsr.get64();
        let v1 = lfsr.get64();
        let v2 = lfsr.get64();
        let v3 = lfsr.get64();

        let mut x = BigInt::<2>::from_parts(&[v0, v1]);
        let y = BigInt::<2>::from_parts(&[v2, v3]);

        let n1 = (v0 as u128) + ((v1 as u128) << 64);
        let n2 = (v2 as u128) + ((v3 as u128) << 64);
        let res1 = n1.overflowing_mul(n2);

        assert_eq!(x.get_part(0), n1 as u64);
        assert_eq!(x.get_part(1), (n1 >> 64) as u64);
        x.dump();
        y.dump();
        let c0 = x.mul::<4>(y);
        x.dump();

        println!("{:x}", res1.0);
        println!("{:x}-{:x}", x.get_part(1), x.get_part(0));

        assert_eq!(x.get_part(0), res1.0 as u64);
        assert_eq!(x.get_part(1), (res1.0 >> 64) as u64);
        assert_eq!(c0, res1.1);
    }
}
