//! This module contains the implementation of string conversion.

extern crate alloc;

use super::bigint::BigInt;
use super::float::Float;
use super::RoundingMode;
use super::Semantics;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::Display;

impl Float {
    /// Convert the number into a large integer, and a base-10 exponent.
    fn convert_to_integer(&self) -> (BigInt, i64) {
        // The natural representation of numbers is 1.mmmmmmm, where the
        // mantissa is aligned to the MSB. In this method we convert the numbers
        // into integers, that start at bit zero, so we use exponent that refers
        //  to bit zero.
        // See Ryu: Fast Float-to-String Conversion -- Ulf Adams.
        // https://youtu.be/kw-U6smcLzk?t=681
        let mut exp = self.get_exp() - self.get_mantissa_len() as i64;
        let mut mantissa: BigInt = self.get_mantissa();

        match exp.cmp(&0) {
            Ordering::Less => {
                // The number is not yet an integer, we need to convert it using
                // the method:
                // mmmmm * 5^(e) * 10 ^(-e) == mmmmm * 10 ^ (-e);
                // where (5^e) * (10^-e) == (2^-e)
                // And the left hand side is how we represent our binary number
                // 1.mmmm * 2^-e, and the right-hand-side is how we represent
                // our decimal number: nnnnnnn * 10^-e.
                let five = BigInt::from_u64(5);
                let e5 = five.powi((-exp) as u64);
                mantissa.inplace_mul(&e5);
                exp = -exp;
            }
            Ordering::Equal | Ordering::Greater => {
                // The number is already an integer, just align it.
                // In this case, E - M > 0, so we are aligning the larger
                // integers, for example [1.mmmm * e^15], in FP16 (where M=10).
                mantissa.shift_left(exp as usize);
                exp = 0;
            }
        }

        (mantissa, exp)
    }

    /// Returns the highest number of decimal digits that are needed for
    /// representing this type accurately.
    pub fn get_decimal_accuracy(&self) -> usize {
        // Matula, David W. “A Formalization of Floating-Point Numeric Base
        // N = 2 + floor(n / log_b(B)) = 2 + floor(n / log(10, 2))
        // We convert from bits to base-10 digits: log(2)/log(10) ==> 59/196.
        // A continuous fraction of 5 iteration gives the ratio.
        2 + (self.get_mantissa_len() * 59) / 196
    }

    /// Reduce a number in the representation mmmmm * e^10, to fewer bits in
    /// 'm', based on the max possible digits in the mantissa.
    fn reduce_printed_integer_length(
        &self,
        integer: &mut BigInt,
        exp: &mut i64,
    ) {
        let bits = integer.msb_index();
        if bits <= self.get_mantissa_len() {
            return;
        };
        let needed_bits = bits - self.get_mantissa_len();
        // We convert from bits to base-10 digits: log(2)/log(10) ==> 59/196.
        // A continuous fraction of 5 iteration gives the ratio.
        let mut digits_to_remove = ((needed_bits * 59) / 196) as i64;

        // Only remove digits after the decimal points.
        if digits_to_remove > *exp {
            digits_to_remove = *exp;
        }
        *exp -= digits_to_remove;
        let ten = BigInt::from_u64(10);
        let divisor = ten.powi(digits_to_remove as u64);
        integer.inplace_div(&divisor);
    }

    fn convert_normal_to_string(&self) -> String {
        // Convert the integer to base-10 integer, and e, the exponent in
        // base 10 (scientific notation).
        let (mut integer, mut e) = self.convert_to_integer();

        // Try to shorten the number.
        self.reduce_printed_integer_length(&mut integer, &mut e);

        // Extract the digits: Div10-Mod10-Div10-Mod10 ....
        let mut buff = Vec::new();
        let digits = integer.to_digits::<10>();
        for d in digits {
            buff.push(char::from_digit(d as u32, 10).unwrap())
        }

        debug_assert!(e >= 0);
        // Add the trailing zeros, and make room to place the point.
        while buff.len() < e as usize {
            buff.insert(0, '0');
        }

        buff.insert(buff.len() - e as usize, '.');
        while !buff.is_empty() && buff[buff.len() - 1] == '0' {
            buff.pop();
        }
        String::from_iter(buff)
    }

    /// Convert the number to a string. This is a simple implementation
    /// that does not take into account rounding during the round-trip of
    /// parsing-printing of the value, or scientific notation, and the minimal
    /// representation of numbers. For all of that that check out the paper:
    /// "How to Print Floating-Point Numbers Accurately" by Steele and White.
    fn convert_to_string(&self) -> String {
        // In order to print decimal digits we need a minimum number of mantissa
        // bits for the conversion. Small floats (such as BF16) don't have
        // enough bits, so we cast to a larger number.
        if self.get_semantics().get_mantissa_len() < 16 {
            use crate::FP32;
            return self.cast(FP32).to_string();
        }

        let result = if self.get_sign() { "-" } else { "" };
        let mut result: String = result.to_string();

        let body: String = match self.get_category() {
            super::float::Category::Infinity => "Inf".to_string(),
            super::float::Category::NaN => "NaN".to_string(),
            super::float::Category::Normal => self.convert_normal_to_string(),
            super::float::Category::Zero => "0.0".to_string(),
        };

        result.push_str(&body);
        result
    }
}
impl Display for Float {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.convert_to_string())
    }
}

impl Display for BigInt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_binary())
    }
}

impl RoundingMode {
    pub fn as_string(&self) -> &str {
        match self {
            RoundingMode::None => "None",
            RoundingMode::NearestTiesToEven => "NearestTiesToEven",
            RoundingMode::NearestTiesToAway => "NearestTiesToAway",
            RoundingMode::Zero => "Zero",
            RoundingMode::Positive => "Positive",
            RoundingMode::Negative => "Negative",
        }
    }
}

impl Display for Semantics {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "(exponent:{} precision:{} rm:{})",
            self.get_exponent_len(),
            self.get_precision(),
            self.get_rounding_mode().as_string()
        )
    }
}

#[cfg(feature = "std")]
mod from {
    use core::fmt::{Debug, Display};
    use std::error::Error;

    use crate::{BigInt, Float, Semantics, FP64};

    impl Float {
        /// Try to construct a Float instance with semantics 'sem' from the
        /// string 'value'. Note that the operation of conversion might lose
        /// precision. If you care about precision you might want to use a
        /// higher precision float and downcast.
        pub fn try_from_str(
            value: &str,
            sem: Semantics,
        ) -> Result<Self, ParseError> {
            // Handle the empty case.
            if value.is_empty() {
                return Err(ParseError(ParseErrorKind::InputEmpty));
            }

            // Handle the plus or minus in front of the number.
            let chars = value.as_bytes();
            let (sign, skip) = if chars[0] == b'-' || chars[0] == b'+' {
                (chars[0] == b'-', 1)
            } else {
                (false, 0)
            };
            let value = &value[skip..];

            // Handle Nan.
            if value.eq_ignore_ascii_case("nan") {
                return Ok(Self::nan(sem, sign));
            }

            // Handle Inf.
            if value.eq_ignore_ascii_case("inf") {
                return Ok(Self::inf(sem, sign));
            }

            // Start handling the non trivial cases.

            let l_r = value.split_once('.');
            // Handle cases where we have no `.` and as such no mantissa.
            if l_r.is_none() {
                // No period. Just parse the integer.
                let ((num, _), exp_num) = parse_with_exp(value)?;
                let mut num = Float::from_bigint(sem, num);

                // Shift the number according to the exponent (in decimal).
                if let Some(exp) = exp_num {
                    if exp >= 0 {
                        num *= Float::from_bigint(
                            sem,
                            BigInt::from_u64(10).powi(exp as u64),
                        );
                    } else {
                        num /= Float::from_bigint(
                            sem,
                            BigInt::from_u64(10).powi((-exp) as u64),
                        );
                    }
                }
                num.set_sign(sign);
                return Ok(num);
            }

            // Handle cases where we have a period in the number:
            let (left, right) = l_r.unwrap();

            // Try parsing decimal value of 0.
            if right.chars().all(|chr| chr == '0') {
                return parse_whole_num(left, sign, sem).map(Ok).unwrap_or(
                    Err(ParseError(ParseErrorKind::ParsingNumberFailed)),
                );
            }

            // Parse the integer part.
            let left_num = parse_big_int(left).map(Ok).unwrap_or(Err(
                ParseError(ParseErrorKind::ParsingNumberFailed),
            ))?;

            // Parse the mantissa and an optional exponent part
            let ((right_num, right_num_digits), explicit_exp) =
                parse_with_exp(right)?;

            // Construct the integral and fractional parts, without the exp.
            // This is one of the places where we might lose precision.
            let dec_shift = BigInt::from_u64(10).powi(right_num_digits as u64);

            let integral = Float::from_bigint(sem, left_num);
            let fraction = Float::from_bigint(sem, right_num)
                / Float::from_bigint(sem, dec_shift);

            // Construct the whole number, move the fractional part into place.
            let mut ret = integral + fraction;

            // Handle the explicit exponent. (Example: e+1).
            if let Some(exp_num) = explicit_exp {
                if exp_num >= 0 {
                    let e = BigInt::from_u64(10).powi(exp_num as u64);
                    ret *= Float::from_bigint(sem, e)
                } else {
                    let e = BigInt::from_u64(10).powi((-exp_num) as u64);
                    ret /= Float::from_bigint(sem, e)
                }
            }
            ret.set_sign(sign);
            Ok(ret)
        }
    }

    impl TryFrom<&str> for Float {
        type Error = ParseError;

        fn try_from(value: &str) -> Result<Self, Self::Error> {
            const DEFAULT_SEM: Semantics = FP64;
            // TODO: autodetect required semantics
            Self::try_from_str(value, DEFAULT_SEM)
        }
    }

    /// Parse a number that contains the 'e' marker for exponent.
    /// Example: 565e+1
    /// Returns the number, the number of decimal digits, and an optional
    /// exponent value.
    fn parse_with_exp(
        value: &str,
    ) -> Result<((BigInt, usize), Option<i64>), ParseError> {
        let idx = value.find(|c| c == 'e' || c == 'E');
        // Split the number to the digits and the exponent.
        let (num_raw, exp) = if let Some(idx) = idx {
            let (l, r) = value.split_at(idx);
            (l, Some(&r[1..]))
        } else {
            (value, None)
        };

        // Parse the left size of the expression (the number).
        let num = parse_big_int(num_raw)
            .map(|num| Ok((num, num_raw.len())))
            .unwrap_or(Err(ParseError(ParseErrorKind::ParsingNumberFailed)))?;

        // Parse the right side (the exponent expression).
        if let Some(exp) = exp {
            match exp.parse::<i64>() {
                Ok(exp) => {
                    // Found a valid expression that has exponent.
                    return Ok((num, Some(exp)));
                }
                Err(_) => {
                    return Err(ParseError(ParseErrorKind::ExponentParseFailed))
                }
            }
        }
        // Found a valid number without exponent marker.
        Ok((num, None))
    }

    /// Try to parse a number from the string 'value'.
    fn parse_big_int(value: &str) -> Option<BigInt> {
        let chars = value.as_bytes();
        let ten = BigInt::from_u64(10);
        let mut num = BigInt::from_u64(0);
        for digit in chars.iter() {
            if *digit > b'9' || *digit < b'0' {
                return None;
            }
            let part = [*digit as u64 - '0' as u64];
            num.inplace_mul(&ten);
            num.inplace_add_slice(&part);
        }
        Some(num)
    }

    /// Parse one long integer and apply a sign.
    fn parse_whole_num(
        value: &str,
        sign: bool,
        sem: Semantics,
    ) -> Option<Float> {
        let chars = value.as_bytes();
        // Handle the special case of '0'.
        if value.len() == 1 && chars[0] == b'0' {
            return Some(Float::zero(sem, sign));
        }

        // Parse the digits.
        let num = parse_big_int(value)?;
        // And construct the Float number.
        let mut ret = Float::from_bigint(sem, num);
        ret.set_sign(sign);

        Some(ret)
    }

    enum ParseErrorKind {
        InputEmpty,
        ParsingNumberFailed,
        ExponentParseFailed,
    }

    pub struct ParseError(ParseErrorKind);

    impl Error for ParseError {}

    impl Display for ParseError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self.0 {
                ParseErrorKind::ParsingNumberFailed => f.write_str(
                    "Failed parsing number part of floating point number",
                ),
                ParseErrorKind::ExponentParseFailed => {
                    f.write_str("Failed parsing exponent of float number")
                }
                ParseErrorKind::InputEmpty => {
                    f.write_str("The input provided was empty")
                }
            }
        }
    }

    impl Debug for ParseError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            Display::fmt(&self, f)
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn test_convert_to_string() {
    use crate::FP16;
    use crate::FP64;
    use core::f64;
    use std::format;

    fn to_str_w_fp16(val: f64) -> String {
        format!("{}", Float::from_f64(val).cast(FP16))
    }

    fn to_str_w_bf16(val: f64) -> String {
        use crate::BF16;
        format!("{}", Float::from_f64(val).cast(BF16))
    }

    fn to_str_w_fp64(val: f64) -> String {
        format!("{}", Float::from_f64(val).cast(FP64))
    }

    assert_eq!("-0.0", to_str_w_fp16(-0.));
    assert_eq!(".30004882", to_str_w_fp16(0.3));
    assert_eq!("4.5", to_str_w_fp16(4.5));
    assert_eq!("256.", to_str_w_fp16(256.));
    assert_eq!("Inf", to_str_w_fp16(65534.));
    assert_eq!("-Inf", to_str_w_fp16(-65534.));
    assert_eq!(".09997558", to_str_w_fp16(0.1));
    assert_eq!(".1", to_str_w_fp64(0.1));
    assert_eq!(".29999999999999998", to_str_w_fp64(0.3));
    assert_eq!("2251799813685248.", to_str_w_fp64((1u64 << 51) as f64));
    assert_eq!("1995.1994999999999", to_str_w_fp64(1995.1995));
    assert_eq!("3.15625", to_str_w_bf16(f64::consts::PI));
}

#[test]
fn test_from_string() {
    assert_eq!("-3.", Float::try_from("-3.0").unwrap().to_string());
    assert_eq!("-3.", Float::try_from("-3.00").unwrap().to_string());
    assert_eq!("30.", Float::try_from("30").unwrap().to_string());
    assert_eq!("430.56", Float::try_from("430.56").unwrap().to_string());
    assert_eq!("5.2", Float::try_from("5.2").unwrap().to_string());
    assert_eq!("Inf", Float::try_from("inf").unwrap().to_string());
    assert_eq!("NaN", Float::try_from("nan").unwrap().to_string());
    assert_eq!("32.", Float::try_from("3.2e1").unwrap().to_string());
    assert_eq!("4.4", Float::try_from("44.e-1").unwrap().to_string());
    assert_eq!("5.4", Float::try_from("54e-1").unwrap().to_string());
    assert_eq!("-5.485", Float::try_from("-54.85e-1").unwrap().to_string());
    assert!(Float::try_from("abc.de").is_err());
    assert!(Float::try_from("e.-21").is_err());
    assert!(Float::try_from("-rlp.").is_err());
    assert!(Float::try_from("").is_err());
}

#[test]
fn test_fuzz_printing() {
    use crate::utils;

    let mut lfsr = utils::Lfsr::new();

    for _ in 0..500 {
        let v0 = lfsr.get64();
        let f0 = f64::from_bits(v0);
        let fp0 = Float::from_f64(f0);
        fp0.to_string();
    }
}

#[cfg(feature = "std")]
#[test]
fn test_print_sqrt() {
    use crate::FP64;
    use std::println;

    // Use Newton-Raphson to find the square root of 5.
    let n = Float::from_u64(FP64, 5);

    let mut x = n.clone();

    for _ in 0..100 {
        x = (&x + (&n / &x)) / 2;
    }
    println!("{}", x);
}

#[test]
#[cfg(feature = "std")]
fn test_readme_example() {
    use std::println;
    // Create a new type: 15 bits exponent, 112 significand.

    // Use Newton-Raphson to find the square root of 5.
    let n = Float::from_u64(FP128, 5);
    let mut x = n.clone();

    for _ in 0..1000 {
        x = (&x + &n / &x) / 2;
    }
    println!("fp128: {}", x);
    println!("fp64:  {}", x.as_f64());

    use crate::{FP128, FP16};
    let fp = Float::from_i64(FP16, 15);
    fp.dump();
}

#[test]
fn test_decimal_accuracy_for_type() {
    use crate::{FP128, FP16, FP256, FP32, FP64};
    assert_eq!(Float::zero(FP16, false).get_decimal_accuracy(), 5);
    assert_eq!(Float::zero(FP32, false).get_decimal_accuracy(), 8);
    assert_eq!(Float::zero(FP64, false).get_decimal_accuracy(), 17);
    assert_eq!(Float::zero(FP128, false).get_decimal_accuracy(), 35);
    assert_eq!(Float::zero(FP256, false).get_decimal_accuracy(), 73);
}

impl BigInt {
    /// Prints the bigint as a decimal number.
    pub fn as_decimal(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }

        let mut buff = Vec::new();
        let digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        let ten = Self::from_u64(10);
        let mut val = self.clone();
        while !val.is_zero() {
            let rem = val.inplace_div(&ten);
            buff.insert(0, digits[rem.as_u64() as usize]);
        }

        String::from_iter(buff)
    }
    /// Prints the bigint as a sequence of bits.
    pub fn as_binary(&self) -> String {
        let mut sb = String::new();

        if self.is_empty() || self.is_zero() {
            return String::from("0");
        }
        let mut top_non_zero = 0;
        for i in (0..self.len()).rev() {
            if self.get_part(i) != 0 {
                top_non_zero = i;
                break;
            }
        }

        for i in 0..=top_non_zero {
            let mut part = self.get_part(i);
            // Don't print leading zeros for the first word.
            if i == top_non_zero {
                while part > 0 {
                    let last = if part & 0x1 == 1 { '1' } else { '0' };
                    sb.insert(0, last);
                    part /= 2;
                }
                continue;
            }

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
}

#[cfg(feature = "std")]
#[test]
fn test_bigint_to_string() {
    let val = 0b101110011010011111010101011110000000101011110101;
    let mut bi = BigInt::from_u64(val);
    bi.shift_left(32);
    assert_eq!(
        bi.as_binary(),
        "10111001101001111101010101111000\
        000010101111010100000000000000000\
        000000000000000"
    );

    let mut bi = BigInt::from_u64(val);
    bi.shift_left(64);
    bi = bi + val;
    assert_eq!(
        bi.as_binary(),
        "101110011010011111010101011110000000101011110101\
         0000000000000000\
         101110011010011111010101011110000000101011110101"
    );
}

#[cfg(feature = "std")]
#[test]
fn test_bigint_to_decimal() {
    let mut num = BigInt::one();
    for i in 1..41 {
        let term = BigInt::from_u64(i);
        num.inplace_mul(&term);
    }

    assert_eq!(
        num.as_decimal(),
        "815915283247897734345611269596115894272000000000"
    );
}
