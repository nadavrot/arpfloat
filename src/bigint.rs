//! This module contains the implementation of the big-int data structure that
//! we use for the significand of the float.

extern crate alloc;

use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign,
};

use alloc::vec::Vec;

/// Reports the kind of values that are lost when we shift right bits. In some
/// context this used as the two guard bits.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LossFraction {
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
    #[allow(dead_code)]
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
/// This is an arbitrary-size unsigned big number implementation. It is used to
/// store the mantissa of the floating point number. The BigInt data structure
/// is backed by `Vec<u64>`, and the data is heap-allocated. BigInt implements
/// the basic arithmetic operations such as add, sub, div, mul, etc.
///
/// # Examples
///
/// ```
///    use arpfloat::BigInt;
///
///    let x = BigInt::from_u64(1995);
///    let y = BigInt::from_u64(90210);
///
///    let z = x * y;
///    let z = z.powi(10);
///
///    // Prints: 3564312949426686000....
///    println!("{}", z.as_decimal());
/// ```
///
#[derive(Debug, Clone)]
pub struct BigInt {
    parts: Vec<u64>,
}

impl BigInt {
    /// Create a new zero big int number.
    pub fn zero() -> Self {
        BigInt::from_u64(0)
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
        let vec = Vec::from([val]);
        BigInt { parts: vec }
    }

    /// Create a number and set the lowest 128 bits to `val`.
    pub fn from_u128(val: u128) -> Self {
        let a = val as u64;
        let b = (val >> 64) as u64;
        let vec = Vec::from([a, b]);
        BigInt { parts: vec }
    }

    /// Create a pseudorandom number with `parts` number of parts in the word.
    /// The random number generator is initialized with `seed`.
    pub fn pseudorandom(parts: usize, seed: u32) -> Self {
        use crate::utils::Lfsr;
        let mut ll = Lfsr::new_with_seed(seed);

        BigInt::from_iter(&mut ll, parts)
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Returns the lowest 64 bits.
    pub fn as_u64(&self) -> u64 {
        for i in 1..self.len() {
            debug_assert_eq!(self.parts[i], 0);
        }
        self.parts[0]
    }

    /// Returns the lowest 64 bits.
    pub fn as_u128(&self) -> u128 {
        if self.len() >= 2 {
            for i in 2..self.len() {
                debug_assert_eq!(self.parts[i], 0);
            }
            (self.parts[0] as u128) + ((self.parts[1] as u128) << 64)
        } else {
            self.parts[0] as u128
        }
    }

    /// Return true if the number is equal to zero.
    pub fn is_zero(&self) -> bool {
        for elem in self.parts.iter() {
            if *elem != 0 {
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
        self.grow(which_word + 1);
        debug_assert!(which_word < self.len(), "Bit out of bounds");
        self.parts[which_word] ^= 1 << bit_in_word;
    }

    /// Zero out all of the bits above `bits`.
    pub fn mask(&mut self, bits: usize) {
        let mut bits = bits;
        for i in 0..self.len() {
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
    pub(crate) fn get_loss_kind_for_bit(&self, bit: usize) -> LossFraction {
        if self.is_zero() {
            return LossFraction::ExactlyZero;
        }
        if bit > self.len() * 64 {
            return LossFraction::LessThanHalf;
        }
        let mut a = self.clone();
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
        for i in (0..self.len()).rev() {
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
        for i in 0..self.len() {
            let part = self.parts[i];
            if part != 0 {
                let idx = part.trailing_zeros() as usize;
                return i * 64 + idx;
            }
        }
        panic!("Expected a non-zero number");
    }

    // Construct a bigint from the words in 'parts'.
    pub fn from_parts(parts: &[u64]) -> Self {
        let parts: Vec<u64> = parts.to_vec();
        BigInt { parts }
    }

    // Construct a bigint from an iterator that generates u64 parts.
    // Take the first 'k' words.
    pub fn from_iter<I: Iterator<Item = u64>>(iter: &mut I, k: usize) -> Self {
        let parts: Vec<u64> = iter.take(k).collect();
        BigInt { parts }
    }

    /// Ensure that there are at least 'size' words in the bigint.
    pub fn grow(&mut self, size: usize) {
        for _ in self.len()..size {
            self.parts.push(0);
        }
    }

    /// Remove the leading zero words from the bigint.
    fn shrink(&mut self) {
        while self.len() > 2 && self.parts[self.len() - 1] == 0 {
            self.parts.pop();
        }
    }

    /// Add `rhs` to this number.
    pub fn inplace_add(&mut self, rhs: &Self) {
        self.inplace_add_slice(&rhs.parts[..]);
    }

    /// Implements addition of the 'rhs' sequence of words to this number.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn inplace_add_slice(&mut self, rhs: &[u64]) {
        self.grow(rhs.len());
        let mut carry: bool = false;
        for i in 0..rhs.len() {
            let first = self.parts[i].overflowing_add(rhs[i]);
            let second = first.0.overflowing_add(carry as u64);
            carry = first.1 || second.1;
            self.parts[i] = second.0;
        }
        // Continue to propagate the carry flag.
        for i in rhs.len()..self.len() {
            let second = self.parts[i].overflowing_add(carry as u64);
            carry = second.1;
            self.parts[i] = second.0;
        }
        if carry {
            self.parts.push(1);
        }
        self.shrink()
    }

    /// Add `rhs` to self, and return true if the operation overflowed (borrow).
    #[must_use]
    pub fn inplace_sub(&mut self, rhs: &Self) -> bool {
        self.inplace_sub_slice(&rhs.parts[..], 0)
    }

    /// Implements subtraction of the 'rhs' sequence of words to this number.
    /// The parameter `known_zeros` specifies how many lower *words* in `rhs`
    /// are zeros and can be ignored. This is used by the division algorithm
    /// that shifts the divisor.
    #[allow(clippy::needless_range_loop)]
    fn inplace_sub_slice(&mut self, rhs: &[u64], bottom_zeros: usize) -> bool {
        self.grow(rhs.len());
        let mut borrow: bool = false;
        // Do the part of the vectors that both sides have.

        for i in bottom_zeros..rhs.len() {
            let first = self.parts[i].overflowing_sub(rhs[i]);
            let second = first.0.overflowing_sub(borrow as u64);
            borrow = first.1 || second.1;
            self.parts[i] = second.0;
        }
        // Propagate the carry bit.
        for i in rhs.len()..self.len() {
            let second = self.parts[i].overflowing_sub(borrow as u64);
            self.parts[i] = second.0;
            borrow = second.1;
        }
        self.shrink();
        borrow
    }

    fn zeros(size: usize) -> Vec<u64> {
        core::iter::repeat(0).take(size).collect()
    }

    /// Multiply `rhs` to self, and return true if the operation overflowed.
    pub fn inplace_mul(&mut self, rhs: &Self) {
        if self.len() > KARATSUBA_SIZE_THRESHOLD
            || rhs.len() > KARATSUBA_SIZE_THRESHOLD
        {
            *self = Self::mul_karatsuba(self, rhs);
            return;
        }
        self.inplace_mul_slice(rhs);
    }

    /// Implements multiplication of the 'rhs' sequence of words to this number.
    fn inplace_mul_slice(&mut self, rhs: &[u64]) {
        let size = self.len() + rhs.len() + 1;
        let mut parts = Self::zeros(size);
        let mut carries = Self::zeros(size);

        for i in 0..self.len() {
            for j in 0..rhs.len() {
                let pi = self.parts[i] as u128;
                let pij = pi * rhs[j] as u128;

                let add0 = parts[i + j].overflowing_add(pij as u64);
                parts[i + j] = add0.0;
                carries[i + j] += add0.1 as u64;
                let add1 = parts[i + j + 1].overflowing_add((pij >> 64) as u64);
                parts[i + j + 1] = add1.0;
                carries[i + j + 1] += add1.1 as u64;
            }
        }
        self.grow(size);
        let mut carry: u64 = 0;
        for i in 0..size {
            let add0 = parts[i].overflowing_add(carry);
            self.parts[i] = add0.0;
            carry = add0.1 as u64 + carries[i];
        }
        self.shrink();
        assert!(carry == 0);
    }

    /// Divide self by `divisor`, and return the reminder.
    pub fn inplace_div(&mut self, divisor: &Self) -> Self {
        let mut dividend = self.clone();
        let mut divisor = divisor.clone();
        let mut quotient = Self::zero();

        // Single word division.
        if self.len() == 1 && divisor.parts.len() == 1 {
            let a = dividend.get_part(0);
            let b = divisor.get_part(0);
            let res = a / b;
            let rem = a % b;
            self.parts[0] = res;
            return Self::from_u64(rem);
        }

        let dividend_msb = dividend.msb_index();
        let divisor_msb = divisor.msb_index();
        assert_ne!(divisor_msb, 0, "division by zero");

        if divisor_msb > dividend_msb {
            let ret = self.clone();
            *self = Self::zero();
            return ret;
        }

        // Align the first bit of the divisor with the first bit of the
        // dividend.
        let bits = dividend_msb - divisor_msb;
        divisor.shift_left(bits);

        // Perform the long division.
        for i in (0..bits + 1).rev() {
            // Find out how many of the lower words of the divisor are zeros.
            let low_zeros = i / 64;

            if dividend >= divisor {
                let overflow = dividend.inplace_sub_slice(&divisor, low_zeros);
                debug_assert!(!overflow);
                quotient.flip_bit(i);
            }
            divisor.shift_right(1);
        }

        *self = quotient;
        self.shrink();
        dividend
    }

    /// Shift the bits in the numbers `bits` to the left.
    pub fn shift_left(&mut self, bits: usize) {
        let words_to_shift = bits / u64::BITS as usize;
        let bits_in_word = bits % u64::BITS as usize;

        for _ in 0..words_to_shift + 1 {
            self.parts.push(0);
        }

        // If we only need to move blocks.
        if bits_in_word == 0 {
            for i in (0..self.len()).rev() {
                self.parts[i] = if i >= words_to_shift {
                    self.parts[i - words_to_shift]
                } else {
                    0
                };
            }
            return;
        }

        for i in (0..self.len()).rev() {
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
            for i in 0..self.len() {
                self.parts[i] = if i + words_to_shift < self.len() {
                    self.parts[i + words_to_shift]
                } else {
                    0
                };
            }
            self.shrink();
            return;
        }

        for i in 0..self.len() {
            let left_val = if i + words_to_shift < self.len() {
                self.parts[i + words_to_shift]
            } else {
                0
            };
            let right_val = if i + 1 + words_to_shift < self.len() {
                self.parts[i + 1 + words_to_shift]
            } else {
                0
            };
            let right = right_val << (u64::BITS as usize - bits_in_word);
            let left = left_val >> bits_in_word;
            self.parts[i] = left | right;
        }
        self.shrink();
    }

    /// Raise this number to the power of `exp` and return the value.
    pub fn powi(&self, mut exp: u64) -> Self {
        let mut v = Self::one();
        let mut base = self.clone();
        loop {
            if exp & 0x1 == 1 {
                v.inplace_mul(&base);
            }
            exp >>= 1;
            if exp == 0 {
                break;
            }
            base.inplace_mul(&base.clone());
        }
        v
    }

    /// Returns the word at idx `idx`.
    pub fn get_part(&self, idx: usize) -> u64 {
        self.parts[idx]
    }

    #[cfg(feature = "std")]
    pub fn dump(&self) {
        use std::println;
        println!("[{}]", self.as_binary());
    }
}

impl Default for BigInt {
    fn default() -> Self {
        Self::zero()
    }
}

#[test]
fn test_powi5() {
    let lookup = [1, 5, 25, 125, 625, 3125, 15625, 78125];
    for (i, val) in lookup.iter().enumerate() {
        let five = BigInt::from_u64(5);
        assert_eq!(five.powi(i as u64).as_u64(), *val);
    }

    // 15 ^ 16
    let v15 = BigInt::from_u64(15);
    assert_eq!(v15.powi(16).as_u64(), 6568408355712890625);

    // 3 ^ 21
    let v3 = BigInt::from_u64(3);
    assert_eq!(v3.powi(21).as_u64(), 10460353203);
}

#[test]
fn test_shl() {
    let mut x = BigInt::from_u64(0xff00ff);
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
    let mut x = BigInt::from_u64(0xff00ff);
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
fn test_mul_basic() {
    let mut x = BigInt::from_u64(0xffff_ffff_ffff_ffff);
    let y = BigInt::from_u64(25);
    x.inplace_mul(&x.clone());
    x.inplace_mul(&y);
    assert_eq!(x.get_part(0), 0x19);
    assert_eq!(x.get_part(1), 0xffff_ffff_ffff_ffce);
    assert_eq!(x.get_part(2), 0x18);
}

#[test]
fn test_add_basic() {
    let mut x = BigInt::from_u64(0xffffffff00000000);
    let y = BigInt::from_u64(0xffffffff);
    let z = BigInt::from_u64(0xf);
    x.inplace_add(&y);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    x.inplace_add(&z);
    assert_eq!(x.get_part(0), 0xe);
    assert_eq!(x.get_part(1), 0x1);
}

#[test]
fn test_div_basic() {
    let mut x1 = BigInt::from_u64(49);
    let mut x2 = BigInt::from_u64(703);
    let y = BigInt::from_u64(7);

    let rem = x1.inplace_div(&y);
    assert_eq!(x1.as_u64(), 7);
    assert_eq!(rem.as_u64(), 0);

    let rem = x2.inplace_div(&y);
    assert_eq!(x2.as_u64(), 100);
    assert_eq!(rem.as_u64(), 3);
}

#[test]
fn test_div_10() {
    let mut x1 = BigInt::from_u64(19940521);
    let ten = BigInt::from_u64(10);
    assert_eq!(x1.inplace_div(&ten).as_u64(), 1);
    assert_eq!(x1.inplace_div(&ten).as_u64(), 2);
    assert_eq!(x1.inplace_div(&ten).as_u64(), 5);
    assert_eq!(x1.inplace_div(&ten).as_u64(), 0);
    assert_eq!(x1.inplace_div(&ten).as_u64(), 4);
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
    let mut x = BigInt::from_parts(&[0x0, 0x1, 0]);
    let y = BigInt::from_u64(0x1);
    let c1 = x.inplace_sub(&y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0xffffffffffffffff);
    assert_eq!(x.get_part(1), 0);

    let mut x = BigInt::from_parts(&[0x1, 0x1]);
    let y = BigInt::from_parts(&[0x0, 0x1, 0x0]);
    let c1 = x.inplace_sub(&y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0x1);
    assert_eq!(x.get_part(1), 0);

    let mut x = BigInt::from_parts(&[0x1, 0x1, 0x1]);
    let y = BigInt::from_parts(&[0x0, 0x1, 0x0]);
    let c1 = x.inplace_sub(&y);
    assert!(!c1);
    assert_eq!(x.get_part(0), 0x1);
    assert_eq!(x.get_part(1), 0);
    assert_eq!(x.get_part(2), 0x1);
}

#[test]
fn test_mask_basic() {
    let mut x = BigInt::from_parts(&[0b11111, 0b10101010101010, 0b111]);
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
        let mut a = BigInt::from_u128(a);
        let b = BigInt::from_u128(b);
        let c = a.inplace_sub(&b);
        (a.as_u128(), c)
    }
    fn test_add(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::from_u128(a);
        let b = BigInt::from_u128(b);
        let mut carry = false;
        a.inplace_add(&b);
        if a.len() > 2 {
            carry = true;
            a.parts[2] = 0;
        }

        (a.as_u128(), carry)
    }
    fn test_mul(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::from_u128(a);
        let b = BigInt::from_u128(b);
        let mut carry = false;
        a.inplace_mul(&b);
        if a.len() > 2 {
            carry = true;
            a.parts[2] = 0;
            a.parts[3] = 0;
        }
        (a.as_u128(), carry)
    }
    fn test_div(a: u128, b: u128) -> (u128, bool) {
        let mut a = BigInt::from_u128(a);
        let b = BigInt::from_u128(b);
        a.inplace_div(&b);
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
        let a = BigInt::from_u128(a);
        let b = BigInt::from_u128(b);

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
    let x = BigInt::from_u64(0xffffffff00000000);
    assert_eq!(x.msb_index(), 64);

    let x = BigInt::from_u64(0x0);
    assert_eq!(x.msb_index(), 0);

    let x = BigInt::from_u64(0x1);
    assert_eq!(x.msb_index(), 1);

    let mut x = BigInt::from_u64(0x1);
    x.shift_left(189);
    assert_eq!(x.msb_index(), 189 + 1);

    for i in 0..256 {
        let mut x = BigInt::from_u64(0x1);
        x.shift_left(i);
        assert_eq!(x.msb_index(), i + 1);
    }
}

#[test]
fn test_trailing_zero() {
    let x = BigInt::from_u64(0xffffffff00000000);
    assert_eq!(x.trailing_zeros(), 32);

    let x = BigInt::from_u64(0x1);
    assert_eq!(x.trailing_zeros(), 0);

    let x = BigInt::from_u64(0x8);
    assert_eq!(x.trailing_zeros(), 3);

    let mut x = BigInt::from_u64(0x1);
    x.shift_left(189);
    assert_eq!(x.trailing_zeros(), 189);

    for i in 0..256 {
        let mut x = BigInt::from_u64(0x1);
        x.shift_left(i);
        assert_eq!(x.trailing_zeros(), i);
    }
}
impl Eq for BigInt {}

impl PartialEq for BigInt {
    fn eq(&self, other: &BigInt) -> bool {
        self.cmp(other).is_eq()
    }
}
impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        // This part word is longer.
        if self.len() > other.len()
            && self.parts[other.len()..].iter().any(|&x| x != 0)
        {
            return Ordering::Greater;
        }

        // The other word is longer.
        if other.len() > self.len()
            && other.parts[self.len()..].iter().any(|&x| x != 0)
        {
            return Ordering::Less;
        }
        let same_len = other.len().min(self.len());

        // Compare all of the digits, from MSB to LSB.
        for i in (0..same_len).rev() {
            match self.parts[i].cmp(&other.parts[i]) {
                Ordering::Less => return Ordering::Less,
                Ordering::Equal => {}
                Ordering::Greater => return Ordering::Greater,
            }
        }
        Ordering::Equal
    }
}

macro_rules! declare_operator {
    ($trait_name:ident,
     $func_name:ident,
     $func_impl_name:ident) => {
        // Self + Self
        impl $trait_name for BigInt {
            type Output = Self;

            fn $func_name(self, rhs: Self) -> Self::Output {
                self.$func_name(&rhs)
            }
        }

        // Self + &Self -> Self
        impl $trait_name<&Self> for BigInt {
            type Output = Self;
            fn $func_name(self, rhs: &Self) -> Self::Output {
                let mut n = self;
                let _ = n.$func_impl_name(rhs);
                n
            }
        }

        // &Self + &Self -> Self
        impl $trait_name<Self> for &BigInt {
            type Output = BigInt;
            fn $func_name(self, rhs: Self) -> Self::Output {
                let mut n = self.clone();
                let _ = n.$func_impl_name(rhs);
                n
            }
        }

        // &Self + u64 -> Self
        impl $trait_name<u64> for BigInt {
            type Output = Self;
            fn $func_name(self, rhs: u64) -> Self::Output {
                let mut n = self;
                let _ = n.$func_impl_name(&Self::from_u64(rhs));
                n
            }
        }
    };
}

declare_operator!(Add, add, inplace_add);
declare_operator!(Sub, sub, inplace_sub);
declare_operator!(Mul, mul, inplace_mul);
declare_operator!(Div, div, inplace_div);

macro_rules! declare_assign_operator {
    ($trait_name:ident,
     $func_name:ident,
     $func_impl_name:ident) => {
        impl $trait_name for BigInt {
            fn $func_name(&mut self, rhs: Self) {
                let _ = self.$func_impl_name(&rhs);
            }
        }

        impl $trait_name<&BigInt> for BigInt {
            fn $func_name(&mut self, rhs: &Self) {
                let _ = self.$func_impl_name(&rhs);
            }
        }
    };
}

declare_assign_operator!(AddAssign, add_assign, inplace_add);
declare_assign_operator!(SubAssign, sub_assign, inplace_sub);
declare_assign_operator!(MulAssign, mul_assign, inplace_mul);
declare_assign_operator!(DivAssign, div_assign, inplace_div);

#[test]
fn test_bigint_operators() {
    type BI = BigInt;
    let x = BI::from_u64(10);
    let y = BI::from_u64(1);

    let c = ((&x - &y) * x) / 2;
    assert_eq!(c.as_u64(), 45);
    assert_eq!((&y + &y).as_u64(), 2);
}

#[test]
fn test_all1s_ctor() {
    type BI = BigInt;
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
    type BI = BigInt;

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
    use alloc::vec::Vec;
    // Take a string of symbols and encode them into one large number.
    const BASE: u64 = 5;
    type BI = BigInt;
    let base = BI::from_u64(BASE);
    let mut bitstream = BI::from_u64(0);
    let mut message: Vec<u64> = Vec::new();

    // We can fit this many digits in the bignum without overflowing.
    // Generate a random message.
    for i in 0..275 {
        message.push(((i + 6) * 17) % BASE);
    }

    // Encode the message.
    for letter in &message {
        let letter = BI::from_u64(*letter);
        bitstream.inplace_mul(&base);
        bitstream.inplace_add(&letter);
    }

    let len = message.len();
    // Decode the message
    for idx in (0..len).rev() {
        let rem = bitstream.inplace_div(&base);
        assert_eq!(message[idx], rem.as_u64());
    }
}

impl BigInt {
    /// Converts this number into a sequence of digits in the range 0..DIGIT.
    /// Use a recursive algorithm to split the number in half, if the number is
    /// too big.
    /// Return the number of digits that were converted.
    fn to_digits_impl<const DIGIT: u8>(
        num: &mut BigInt,
        num_digits: usize,
        output: &mut Vec<u8>,
    ) -> usize {
        const SPLIT_WORD_THRESHOLD: usize = 5;

        // Figure out how many digits fit in a single word.
        let bits_per_digit = (8 - DIGIT.leading_zeros()) as usize;
        let digits_per_word = 64 / bits_per_digit;
        let digit = DIGIT as u64;

        // If the word is too big, split it in half.
        let len = num.len();
        if len > SPLIT_WORD_THRESHOLD {
            let half = len / 2 - 1;
            // Figure out how many digits to extract:
            let k = digits_per_word * half;
            // Create a mega digit (a*a*a*a....).
            let mega_digit = BigInt::from_u64(digit).powi(k as u64);
            // Extract the lowest k digits.
            let mut rem = num.inplace_div(&mega_digit);

            // Convert the two parts to digits:
            let tail = Self::to_digits_impl::<DIGIT>(&mut rem, k, output);
            let hd = Self::to_digits_impl::<DIGIT>(num, num_digits - k, output);
            debug_assert_eq!(tail, k);
            debug_assert_eq!(hd, num_digits - k);
            return num_digits;
        }

        let mut extracted = 0;

        // Multiply a*a*a*a ... until we fill a 64bit word.
        let divisor = BigInt::from_u64(digit.pow(digits_per_word as u32));
        // For each word:
        for _ in 0..(num_digits / digits_per_word) {
            // Pull a single word of [a*a*a*a ....].
            let mut rem = num.inplace_div(&divisor);
            // This is fast because we operate on a single word.
            extracted += digits_per_word;
            Self::extract_digits::<DIGIT>(digits_per_word, &mut rem, output);
        }

        // Handle the rest of the digits.
        let iters = num_digits % digits_per_word;
        Self::extract_digits::<DIGIT>(iters, num, output);
        extracted += iters;

        extracted
    }

    // Extract 'iter' digits from 'num', one by one, and push them to 'vec'.
    fn extract_digits<const DIGIT: u8>(
        iter: usize,
        num: &mut BigInt,
        vec: &mut Vec<u8>,
    ) {
        let digit = BigInt::from_u64(DIGIT as u64);
        for _ in 0..iter {
            let d = num.inplace_div(&digit).as_u64();
            vec.push(d as u8);
        }
    }

    /// Converts this number into a sequence of digits in the range 0..DIGIT.
    pub(crate) fn to_digits<const DIGIT: u8>(&self) -> Vec<u8> {
        let mut num = self.clone();
        num.shrink();

        let mut output: Vec<u8> = Vec::new();

        while !num.is_zero() {
            let len = num.len();
            // Figure out how many digits fit in the number.
            // See 'get_decimal_accuracy'.
            let digits = (len * 64 * 59) / 196;
            Self::to_digits_impl::<DIGIT>(&mut num, digits, &mut output);
        }

        // Eliminate leading zeros.

        while output.len() > 1 && output[output.len() - 1] == 0 {
            output.pop();
        }
        output.reverse();
        output
    }
}

#[test]
pub fn test_bigint_to_digits() {
    use alloc::string::String;
    use core::primitive::char;
    /// Convert the vector of digits 'vec' of base 'base' into a string.
    fn vec_to_string(vec: Vec<u8>, base: u32) -> String {
        let mut sb = String::new();
        for d in vec {
            sb.push(char::from_digit(d as u32, base).unwrap())
        }
        sb
    }

    // Test binary.
    let mut num = BigInt::from_u64(0b111000111000101010);
    num.shift_left(64);
    let digits = num.to_digits::<2>();
    assert_eq!(
        vec_to_string(digits, 2),
        "1110001110001010100000000000000\
        0000000000000000000000000000000\
        00000000000000000000"
    );

    // Test base 10.
    let num = BigInt::from_u64(90210);
    let digits = num.to_digits::<10>();
    assert_eq!(vec_to_string(digits, 10), "90210");

    // Test base 10 long.
    let num = BigInt::from_u128(123_456_123_456_987_654_987_654u128);
    let digits = num.to_digits::<10>();
    assert_eq!(vec_to_string(digits, 10), "123456123456987654987654");
}

/// Bigint numbers above this size use the karatsuba algorithm for
/// multiplication. The number represents the number of words in the bigint.
/// Numbers below this threshold use the traditional O(n^2) multiplication.
const KARATSUBA_SIZE_THRESHOLD: usize = 64;

impl BigInt {
    fn mul_karatsuba(lhs: &[u64], rhs: &[u64]) -> BigInt {
        // Algorithm description:
        // https://en.wikipedia.org/wiki/Karatsuba_algorithm

        // Handle small numbers using the traditional O(n^2) algorithm.
        if lhs.len().min(rhs.len()) < KARATSUBA_SIZE_THRESHOLD {
            // Handle zero-sized inputs.
            if lhs.is_empty() || rhs.is_empty() {
                return BigInt::zero();
            }
            let mut lhs = BigInt::from_parts(lhs);
            lhs.inplace_mul_slice(rhs);
            return lhs;
        }

        // Split the big-int into two parts. One of the parts might be
        // zero-sized.
        let mid = lhs.len().max(rhs.len()) / 2;
        let a = &lhs[0..mid.min(lhs.len())];
        let b = &lhs[mid.min(lhs.len())..];
        let c = &rhs[0..mid.min(rhs.len())];
        let d = &rhs[mid.min(rhs.len())..];

        // Compute 'a*c' and 'b*d'.
        let ac = Self::mul_karatsuba(a, c);
        let mut bd = Self::mul_karatsuba(b, d);

        // Compute (a+b) * (c+d).
        let mut a_b = BigInt::from_parts(a);
        a_b.inplace_add_slice(b);
        let mut c_d = BigInt::from_parts(c);
        c_d.inplace_add_slice(d);

        let mut ad_plus_bc = Self::mul_karatsuba(&a_b, &c_d);

        // Compute (a+b) * (c+d) - ac - bd
        ad_plus_bc.inplace_sub_slice(&ac, 0);
        ad_plus_bc.inplace_sub_slice(&bd, 0);

        // Add the parts of the word together.
        bd.shift_left(64 * mid * 2);
        ad_plus_bc.shift_left(64 * mid);
        bd.inplace_add(&ad_plus_bc);
        bd.inplace_add(&ac);
        bd
    }
}

#[test]
fn test_mul_karatsuba() {
    use crate::utils::Lfsr;
    let mut ll = Lfsr::new();

    // Compare the multiplication of karatsuba to the direct multiplication on
    // two random numbers of lengths 'r' and 'l'.
    fn test_sizes(l: usize, r: usize, ll: &mut Lfsr) {
        let mut a = BigInt::from_iter(ll, l);
        let b = BigInt::from_iter(ll, r);
        let res = BigInt::mul_karatsuba(&a, &b);
        a.inplace_mul_slice(&b);
        assert_eq!(res, a);
    }

    test_sizes(1, 1, &mut ll);
    test_sizes(100, 1, &mut ll);
    test_sizes(1, 100, &mut ll);
    test_sizes(100, 100, &mut ll);
    test_sizes(1000, 1000, &mut ll);
    test_sizes(1000, 1001, &mut ll);

    // Try numbers of different sizes.
    for i in 64..90 {
        for j in 1..128 {
            test_sizes(i, j, &mut ll);
        }
    }
}

use core::ops::Deref;

impl Deref for BigInt {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        &self.parts[..]
    }
}
