extern crate alloc;

use alloc::string::String;
#[cfg(test)]
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Sub};

/// Reports the kind of values that are lost when we shift right bits. In some
/// context this used as the two guard bits.
#[derive(Debug, Clone, Copy)]
pub enum LossFraction {
    ExactlyZero,  //0000000
    LessThanHalf, //0xxxxxx
    ExactlyHalf,  //1000000
    MoreThanHalf, //1xxxxxx
}

impl LossFraction {
    pub fn is_exactly_zero(&self) -> bool {
        matches!(self, Self::ExactlyZero)
    }
    pub fn is_lt_half(&self) -> bool {
        matches!(self, Self::LessThanHalf) || self.is_exactly_zero()
    }
    pub fn is_exactly_half(&self) -> bool {
        matches!(self, Self::ExactlyHalf)
    }
    pub fn is_mt_half(&self) -> bool {
        matches!(self, Self::MoreThanHalf)
    }
    pub fn is_lte_half(&self) -> bool {
        self.is_lt_half() || self.is_exactly_half()
    }
    pub fn is_gte_half(&self) -> bool {
        self.is_mt_half() || self.is_exactly_half()
    }

    // Return the inverted loss fraction.
    pub fn invert(&self) -> LossFraction {
        match self {
            LossFraction::LessThanHalf => LossFraction::MoreThanHalf,
            LossFraction::MoreThanHalf => LossFraction::LessThanHalf,
            _ => *self,
        }
    }
}
/// This is a fixed-size big int implementation that's used to represent the
/// significand part of the floating point number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BigInt<const PARTS: usize> {
    parts: [u64; PARTS],
}

impl<const PARTS: usize> BigInt<PARTS> {
    /// Create a new zero big int number.
    pub fn zero() -> Self {
        BigInt { parts: [0; PARTS] }
    }

    /// Create a new number with the value 1.
    pub fn one() -> Self {
        Self::from_u64(1)
    }

    /// Create a new number with a single '1' set at bit `bit`.
    pub fn one_hot(bit: usize) -> Self {
        let mut x = Self::zero();
        x.flip_bit(bit);
        x
    }

    /// Create a new number, where the first `bits` bits are set to 1.
    pub fn all1s(bits: usize) -> Self {
        if bits == 0 {
            return Self::zero();
        }
        let mut x = Self::one();
        x.shift_left(bits);
        let _ = x.inplace_sub(&Self::one());
        debug_assert_eq!(x.msb_index(), bits);
        x
    }

    /// Create a number and set the lowest 64 bits to `val`.
    pub fn from_u64(val: u64) -> Self {
        let mut bi = BigInt { parts: [0; PARTS] };
        bi.parts[0] = val;
        bi
    }

    /// Create a number and set the lowest 128 bits to `val`.
    pub fn from_u128(val: u128) -> Self {
        let mut bi = BigInt { parts: [0; PARTS] };
        bi.parts[0] = val as u64;
        bi.parts[1] = (val >> 64) as u64;
        bi
    }

    /// Returns the lowest 64 bits.
    pub fn as_u64(&self) -> u64 {
        for i in 1..PARTS {
            debug_assert_eq!(self.parts[i], 0);
        }
        self.parts[0]
    }

    /// Returns the lowest 64 bits.
    pub fn as_u128(&self) -> u128 {
        if PARTS >= 2 {
            for i in 2..PARTS {
                debug_assert_eq!(self.parts[i], 0);
            }
            (self.parts[0] as u128) + ((self.parts[1] as u128) << 64)
        } else {
            self.parts[0] as u128
        }
    }

    /// Prints the bigint as a sequence of bits.
    pub fn as_str(&self) -> String {
        let mut sb = String::new();
        let mut first = true;
        for i in (0..PARTS).rev() {
            let mut part = self.parts[i];
            // Don't print leading zeros in empty parts of the bigint.
            if first && part == 0 {
                continue;
            }

            // Don't print leading zeros for the first word.
            if first {
                while part > 0 {
                    let last = if part & 0x1 == 1 { '1' } else { '0' };
                    sb.insert(0, last);
                    part /= 2;
                }
                continue;
            }
            first = false;

            // Print leading zeros for the rest of the words.
            for _ in 0..64 {
                let last = if part & 0x1 == 1 { '1' } else { '0' };
                sb.insert(0, last);
                part /= 2;
            }
        }
        if sb.is_empty() {
            sb.push('0');
        }
        sb
    }

    /// Convert this instance to a smaller number. Notice that this may truncate
    /// the number.
    pub fn cast<const P: usize>(&self) -> BigInt<P> {
        let mut n = BigInt::<P>::zero();
        let to = PARTS.min(P);
        for i in to..PARTS {
            debug_assert_eq!(self.parts[i], 0, "losing information");
        }
        for i in 0..to {
            n.parts[i] = self.parts[i];
        }
        n
    }

    /// \return true if the number is equal to zero.
    pub fn is_zero(&self) -> bool {
        for elem in self.parts {
            if elem != 0 {
                return false;
            }
        }
        true
    }

    /// Returns true if this number is even.
    pub fn is_even(&self) -> bool {
        (self.parts[0] & 0x1) == 0
    }

    /// Returns true if this number is odd.
    pub fn is_odd(&self) -> bool {
        (self.parts[0] & 0x1) == 1
    }

    /// Flip the `bit_num` bit.
    pub fn flip_bit(&mut self, bit_num: usize) {
        let which_word = bit_num / u64::BITS as usize;
        let bit_in_word = bit_num % u64::BITS as usize;
        debug_assert!(which_word < PARTS, "Bit out of bounds");
        self.parts[which_word] ^= 1 << bit_in_word;
    }

    /// Zero out all of the bits above `bits`.
    pub fn mask(&mut self, bits: usize) {
        let mut bits = bits;
        for i in 0..PARTS {
            if bits >= 64 {
                bits -= 64;
                continue;
            }

            if bits == 0 {
                self.parts[i] = 0;
                continue;
            }

            let mask = (1u64 << bits) - 1;
            self.parts[i] &= mask;
            bits = 0;
        }
    }

    /// Returns the fractional part that's lost during truncation at `bit`.
    pub fn get_loss_kind_for_bit(&self, bit: usize) -> LossFraction {
        if self.is_zero() {
            return LossFraction::ExactlyZero;
        }
        if bit > PARTS * 64 {
            return LossFraction::LessThanHalf;
        }
        let mut a = *self;
        a.mask(bit);
        if a.is_zero() {
            return LossFraction::ExactlyZero;
        }
        let half = Self::one_hot(bit - 1);
        match a.cmp(&half) {
            Ordering::Less => LossFraction::LessThanHalf,
            Ordering::Equal => LossFraction::ExactlyHalf,
            Ordering::Greater => LossFraction::MoreThanHalf,
        }
    }

    /// Returns the index of the most significant bit (the highest '1'),
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

    /// Returns the index of the first '1' in the number. The number must not
    ///  be a zero.
    pub fn trailing_zeros(&self) -> usize {
        debug_assert!(!self.is_zero());
        for i in 0..PARTS {
            let part = self.parts[i];
            if part != 0 {
                let idx = part.trailing_zeros() as usize;
                return i * 64 + idx;
            }
        }
        panic!("Expected a non-zero number");
    }

    pub fn from_parts(parts: &[u64; PARTS]) -> Self {
        BigInt { parts: *parts }
    }

    /// Add `rhs` to self, and return true if the operation overflowed.
    #[must_use]
    pub fn inplace_add(&mut self, rhs: &Self) -> bool {
        let mut carry: bool = false;
        for i in 0..PARTS {
            let first = self.parts[i].overflowing_add(rhs.parts[i]);
            let second = first.0.overflowing_add(carry as u64);
            carry = first.1 || second.1;
            self.parts[i] = second.0;
        }
        carry
    }

    /// Add `rhs` to self, and return true if the operation overflowed (borrow).
    #[must_use]
    pub fn inplace_sub(&mut self, rhs: &Self) -> bool {
        let mut borrow: bool = false;
        for i in 0..PARTS {
            let first = self.parts[i].overflowing_sub(rhs.parts[i]);
            let second = first.0.overflowing_sub(borrow as u64);
            borrow = first.1 || second.1;
            self.parts[i] = second.0;
        }
        borrow
    }

    /// Multiply `rhs` to self, and return true if the operation overflowed.
    #[must_use]
    pub fn inplace_mul(&mut self, rhs: Self) -> bool {
        /// The parameter `P2` is here to work around a limitation in the
        /// rust generic system. P2 needs to be greater or equal to PARTS*2.
        const P2: usize = 100;
        debug_assert!(P2 >= PARTS * 2);
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

    /// Divide self by `divisor`, and return the reminder.
    pub fn inplace_div(&mut self, divisor: Self) -> Self {
        let mut dividend = *self;
        let mut divisor = divisor;
        let mut quotient = Self::zero();

        let dividend_msb = dividend.msb_index();
        let divisor_msb = divisor.msb_index();
        assert_ne!(divisor_msb, 0, "division by zero");

        if divisor_msb > dividend_msb {
            let ret = *self;
            *self = Self::zero();
            return ret;
        }

        // Single word division.
        if divisor_msb < 65 && dividend_msb < 65 {
            let a = dividend.get_part(0);
            let b = divisor.get_part(0);
            let res = a / b;
            let rem = a % b;
            self.parts[0] = res;
            return Self::from_u64(rem);
        }

        // This is a fast path for the case where we know that the active bits
        // in the word are smaller than the current size. In this case we call
        // the implementation that uses fewer parts.
        macro_rules! delegate_small_div {
            ($num_parts:expr) => {
                let bigint_size_in_bits = $num_parts * 64;
                if PARTS > $num_parts
                    && dividend_msb < bigint_size_in_bits
                    && divisor_msb < bigint_size_in_bits
                {
                    let mut a4: BigInt<$num_parts> = dividend.cast();
                    let b4: BigInt<$num_parts> = divisor.cast();
                    let rem = a4.inplace_div(b4);
                    *self = a4.cast();
                    return rem.cast();
                }
            };
        }
        delegate_small_div!(2);
        delegate_small_div!(4);
        delegate_small_div!(8);
        delegate_small_div!(16);
        delegate_small_div!(32);
        delegate_small_div!(64);

        // Align the first bit of the divisor with the first bit of the
        // dividend.
        let bits = dividend_msb - divisor_msb;
        divisor.shift_left(bits);

        // Perform the long division.
        for i in (0..bits + 1).rev() {
            if dividend >= divisor {
                dividend = dividend - divisor;
                quotient.flip_bit(i);
            }
            divisor.shift_right(1);
        }

        *self = quotient;
        dividend
    }

    /// Shift the bits in the numbers `bits` to the left.
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

    /// Shift the bits in the numbers `bits` to the right.
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

    /// \return raise this number to the power of `exp`.
    pub fn powi(&self, mut exp: u64) -> Self {
        let mut v = Self::one();
        let mut base = *self;
        loop {
            if exp & 0x1 == 1 {
                let overflow = v.inplace_mul(base);
                debug_assert!(!overflow)
            }
            exp >>= 1;
            if exp == 0 {
                break;
            }
            let overflow = base.inplace_mul(base);
            debug_assert!(!overflow)
        }
        v
    }

    /// \return the word at idx `idx`.
    pub fn get_part(&self, idx: usize) -> u64 {
        self.parts[idx]
    }

    #[cfg(feature = "std")]
    pub fn dump(&self) {
        use std::{print, println};
        print!("[");
        for i in (0..PARTS).rev() {
            let width = u64::BITS as usize;
            print!("|{:0width$b}", self.parts[i]);
        }
        println!("]");
    }
}

impl<const PARTS: usize> Default for BigInt<PARTS> {
    fn default() -> Self {
        Self::zero()
    }
}

#[test]
fn test_powi5() {
    let lookup = [1, 5, 25, 125, 625, 3125, 15625, 78125];
    for (i, val) in lookup.iter().enumerate() {
        let five = BigInt::<4>::from_u64(5);
        assert_eq!(five.powi(i as u64).as_u64(), *val);
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
    assert_eq!(x.get_part(1), 0x807f800000000000);
    x.shift_right(17);
    assert_eq!(x.get_part(1), 0x03fc03fc0000000);
    x.shift_right(64);
    assert_eq!(x.get_part(0), 0x03fc03fc0000000);
}

#[test]
fn test_add_basic() {
    let mut x = BigInt::<2>::from_u64(0xffffffff00000000);
    let y = BigInt::<2>::from_u64(0xffffffff);
    let z = BigInt::<2>::from_u64(0xf);
    let c1 = x.inplace_add(&y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    let c2 = x.inplace_add(&z);
    assert!(!c2);
    assert_eq!(x.get_part(0), 0xe);
    assert_eq!(x.get_part(1), 0x1);
}

#[test]
fn test_div_basic() {
    let mut x1 = BigInt::<2>::from_u64(49);
    let mut x2 = BigInt::<2>::from_u64(703);
    let y = BigInt::<2>::from_u64(7);

    let rem = x1.inplace_div(y);
    assert_eq!(x1.as_u64(), 7);
    assert_eq!(rem.as_u64(), 0);

    let rem = x2.inplace_div(y);
    assert_eq!(x2.as_u64(), 100);
    assert_eq!(rem.as_u64(), 3);
}

#[test]
fn test_div_10() {
    let mut x1 = BigInt::<2>::from_u64(19940521);
    let ten = BigInt::<2>::from_u64(10);
    assert_eq!(x1.inplace_div(ten).as_u64(), 1);
    assert_eq!(x1.inplace_div(ten).as_u64(), 2);
    assert_eq!(x1.inplace_div(ten).as_u64(), 5);
    assert_eq!(x1.inplace_div(ten).as_u64(), 0);
    assert_eq!(x1.inplace_div(ten).as_u64(), 4);
}

#[allow(dead_code)]
fn test_with_random_values(
    correct: fn(u128, u128) -> (u128, bool),
    test: fn(u128, u128) -> (u128, bool),
) {
    use super::utils::Lfsr;

    // Test addition, multiplication, subtraction with random values.
    let mut lfsr = Lfsr::new();

    for _ in 0..50000 {
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
    // Check a single overflowing sub operation.
    let mut x = BigInt::<2>::from_parts(&[0x0, 0x1]);
    let y = BigInt::<2>::from_u64(0x1);
    let c1 = x.inplace_sub(&y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    assert_eq!(x.get_part(1), 0);
}

#[test]
fn test_mask_basic() {
    let mut x = BigInt::<3>::from_parts(&[0b11111, 0b10101010101010, 0b111]);
    x.mask(69);
    assert_eq!(x.get_part(0), 0b11111); // No change
    assert_eq!(x.get_part(1), 0b01010); // Keep the bottom 5 bits.
    assert_eq!(x.get_part(2), 0b0); // Zero.
}

#[test]
fn test_basic_operations() {
    // Check Add, Mul, Sub, in comparison to the double implementation.

    fn correct_sub(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_sub(b)
    }
    fn correct_add(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_add(b)
    }
    fn correct_mul(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_mul(b)
    }
    fn correct_div(a: u128, b: u128) -> (u128, bool) {
        a.overflowing_div(b)
    }

    fn test_sub(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        let c = a.inplace_sub(&b);
        (a.as_u128(), c)
    }
    fn test_add(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        let c = a.inplace_add(&b);
        (a.as_u128(), c)
    }
    fn test_mul(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        let c = a.inplace_mul(b);
        (a.as_u128(), c)
    }
    fn test_div(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);
        a.inplace_div(b);
        (a.as_u128(), false)
    }

    fn correct_cmp(a: u128, b: u128) -> (u128, bool) {
        (
            match a.cmp(&b) {
                Ordering::Less => 1,
                Ordering::Equal => 2,
                Ordering::Greater => 3,
            } as u128,
            false,
        )
    }
    fn test_cmp(a: u128, b: u128) -> (u128, bool) {
        let a = BigInt::<2>::from_u128(a);
        let b = BigInt::<2>::from_u128(b);

        (
            match a.cmp(&b) {
                Ordering::Less => 1,
                Ordering::Equal => 2,
                Ordering::Greater => 3,
            } as u128,
            false,
        )
    }

    test_with_random_values(correct_mul, test_mul);
    test_with_random_values(correct_div, test_div);
    test_with_random_values(correct_add, test_add);
    test_with_random_values(correct_sub, test_sub);
    test_with_random_values(correct_cmp, test_cmp);
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

#[test]
fn test_trailing_zero() {
    let x = BigInt::<5>::from_u64(0xffffffff00000000);
    assert_eq!(x.trailing_zeros(), 32);

    let x = BigInt::<5>::from_u64(0x1);
    assert_eq!(x.trailing_zeros(), 0);

    let x = BigInt::<5>::from_u64(0x8);
    assert_eq!(x.trailing_zeros(), 3);

    let mut x = BigInt::<5>::from_u64(0x1);
    x.shift_left(189);
    assert_eq!(x.trailing_zeros(), 189);

    for i in 0..256 {
        let mut x = BigInt::<5>::from_u64(0x1);
        x.shift_left(i);
        assert_eq!(x.trailing_zeros(), i);
    }
}

impl<const PARTS: usize> PartialOrd for BigInt<PARTS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PARTS: usize> Ord for BigInt<PARTS> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare all of the digits, from MSB to LSB.
        for i in (0..PARTS).rev() {
            match self.parts[i].cmp(&other.parts[i]) {
                Ordering::Less => return Ordering::Less,
                Ordering::Equal => {}
                Ordering::Greater => return Ordering::Greater,
            }
        }
        Ordering::Equal
    }
}

impl<const PARTS: usize> Add for BigInt<PARTS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut n = self;
        let _ = n.inplace_add(&rhs);
        n
    }
}
impl<const PARTS: usize> Sub for BigInt<PARTS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut n = self;
        let _ = n.inplace_sub(&rhs);
        n
    }
}
impl<const PARTS: usize> Mul for BigInt<PARTS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut n = self;
        let overflow = n.inplace_mul(rhs);
        debug_assert!(!overflow);
        n
    }
}
impl<const PARTS: usize> Div for BigInt<PARTS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut n = self;
        n.inplace_div(rhs);
        n
    }
}

#[test]
fn test_bigint_operators() {
    type BI = BigInt<2>;
    let x = BI::from_u64(10);
    let y = BI::from_u64(1);
    let two = BI::from_u64(2);

    let c = ((x - y) * x) / two;
    assert_eq!(c.as_u64(), 45);
    assert_eq!((y + y).as_u64(), 2);
}

#[test]
fn test_all1s_ctor() {
    type BI = BigInt<2>;
    let v0 = BI::all1s(0);
    let v1 = BI::all1s(1);
    let v2 = BI::all1s(5);
    let v3 = BI::all1s(32);

    assert_eq!(v0.get_part(0), 0b0);
    assert_eq!(v1.get_part(0), 0b1);
    assert_eq!(v2.get_part(0), 0b11111);
    assert_eq!(v3.get_part(0), 0xffffffff);
}

#[test]
fn test_flip_bit() {
    type BI = BigInt<2>;

    {
        let mut v0 = BI::zero();
        assert_eq!(v0.get_part(0), 0);
        v0.flip_bit(0);
        assert_eq!(v0.get_part(0), 1);
        v0.flip_bit(0);
        assert_eq!(v0.get_part(0), 0);
    }

    {
        let mut v0 = BI::zero();
        v0.flip_bit(16);
        assert_eq!(v0.get_part(0), 65536);
    }

    {
        let mut v0 = BI::zero();
        v0.flip_bit(95);
        v0.shift_right(95);
        assert_eq!(v0.get_part(0), 1);
    }
}

#[cfg(feature = "std")]
#[test]
fn test_mul_div_encode_decode() {
    // Take a string of symbols and encode them into one large number.
    const P: usize = 10;
    const BASE: u64 = 5;
    type BI = BigInt<P>;
    let base = BI::from_u64(BASE);
    let mut bitstream = BI::from_u64(0);
    let mut message: Vec<u64> = Vec::new();

    // We can fit this many digits in the bignum without overflowing.
    for i in 0..275 {
        message.push(((i + 6) * 17) % BASE);
    }

    // Encode the message.
    for letter in &message {
        let letter = BI::from_u64(*letter);
        let overflow = bitstream.inplace_mul(base);
        assert!(!overflow);
        let overflow = bitstream.inplace_add(&letter);
        assert!(!overflow);
    }

    let len = message.len();
    // Decode the message
    for idx in (0..len).rev() {
        let rem = bitstream.inplace_div(base);
        assert_eq!(message[idx], rem.as_u64());
    }
}
