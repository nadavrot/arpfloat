type PartTy = u64;

#[derive(Debug, Clone, Copy)]
pub struct BigInt<const PARTS: usize> {
    parts: [PartTy; PARTS],
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

    pub fn from_parts(parts: &[PartTy; PARTS]) -> Self {
        BigInt {
            parts: parts.clone(),
        }
    }

    // Add \p rhs to self, and return true if the operation overflowed.
    pub fn add(&mut self, rhs: Self) -> bool {
        let mut carry: bool = false;
        for i in 0..PARTS {
            let first = self.parts[i].overflowing_add(rhs.parts[i]);
            let second =
                first.0.overflowing_add(if carry { 1u64 } else { 0u64 });
            carry = first.1 || second.1;
            self.parts[i] = second.0;
        }
        carry
    }

    // Shift the bits in the numbers \p bits to the left.
    pub fn shift_left(&mut self, bits: usize) {
        let words_to_shift = bits / PartTy::BITS as usize;
        let bits_in_word = bits % PartTy::BITS as usize;

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
            let right = right_val >> (PartTy::BITS as usize - bits_in_word);
            let left = left_val << bits_in_word;
            self.parts[i] = left | right;
        }
    }

    // Shift the bits in the numbers \p bits to the right.
    pub fn shift_right(&mut self, bits: usize) {
        let words_to_shift = bits / PartTy::BITS as usize;
        let bits_in_word = bits % PartTy::BITS as usize;

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
            let right = right_val << (PartTy::BITS as usize - bits_in_word);
            let left = left_val >> bits_in_word;
            self.parts[i] = left | right;
        }
    }

    fn get_part(&self, idx: usize) -> PartTy {
        self.parts[idx]
    }

    fn dump(&self) {
        print!("[");
        for i in (0..PARTS).rev() {
            let width = PartTy::BITS as usize;
            print!("|{:0width$x}", self.parts[i]);
        }
        println!("]");
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
    assert_eq!(c1, false);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    x.dump();
    let c2 = x.add(z);
    assert_eq!(c2, false);
    assert_eq!(x.get_part(0), 0xe);
    assert_eq!(x.get_part(1), 0x1);
    x.dump();
}
