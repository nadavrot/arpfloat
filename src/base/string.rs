use crate::base::FP16;

use super::bigint::BigInt;
use super::float::Float;
use std::fmt::Display;

// Use a bigint for the decimal conversions.
type BigNum = BigInt<6>;

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    /// Convert the number into a large integer, and a base-10 exponent.
    fn convert_to_integer(&self) -> (BigNum, i64) {
        // The natural representation of numbers is 1.mmmmmmm, where the
        // mantissa is aligned to the MSB. In this method we convert the numbers
        // into integers, that start at bit zero, so we use exponent that refers
        //  to bit zero.
        // See Ryu: Fast Float-to-String Conversion -- Ulf Adams.
        // https://youtu.be/kw-U6smcLzk?t=681
        let mut exp = self.get_exp() - MANTISSA as i64;
        let mut mantissa: BigNum = self.get_mantissa().cast();

        match exp.cmp(&0) {
            std::cmp::Ordering::Less => {
                // The number is not yet an integer, we need to convert it using
                // the method:
                // mmmmm * 5^(e) * 10 ^(-e) == mmmmm * 10 ^ (-e);
                // where (5^e) * (10^-e) == (2^-e)
                // And the left hand side is how we represent our binary number
                // 1.mmmm * 2^-e, and the right-hand-side is how we represent
                // our decimal number: nnnnnnn * 10^-e.
                let five = BigInt::from_u64(5);
                let e5 = five.powi((-exp) as u64);
                mantissa.inplace_mul::<12>(e5);
                exp = -exp;
            }
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => {
                // The number is already an integer, just align it.
                // In this case, E - M > 0, so we are aligning the larger
                // integers, for example [1.mmmm * e^15], in FP16 (where M=10).
                mantissa.shift_left(exp as usize);
                exp = 0;
            }
        }

        (mantissa, exp)
    }

    /// Reduce a number in the representation mmmmm * e^10, to fewer bits in
    /// 'm', based on the max possible digits in the mantissa.
    fn reduce_printed_integer_length(integer: &mut BigNum, exp: &mut i64) {
        // Matula, David W. “A Formalization of Floating-Point Numeric Base
        // N = 2 + floor(n / log_b(B)) = 2 + floor(n / log(10, 2))
        // We convert from bits to base-10 digits: log(2)/log(10) ==> 59/196.
        // A continuous fraction of 5 iteration gives the ratio.
        let bits = integer.msb_index();
        if bits <= MANTISSA {
            return;
        };
        let needed_bits = bits - MANTISSA;
        let mut digits_to_remove = ((needed_bits * 59) / 196) as i64;

        // Only remove digits after the decimal points.
        if digits_to_remove > *exp {
            digits_to_remove = *exp;
        }
        *exp -= digits_to_remove;
        let ten = BigInt::from_u64(10);
        let divisor = ten.powi(digits_to_remove as u64);
        integer.inplace_div(divisor);
    }

    /// This method converts floats to strings. This is a simple implementation
    /// that does not take into account rounding during the round-trip of
    /// parsing-printing of the value, or scientific notation, and the minimal
    /// representation of numbers. For all of that that check out the paper:
    /// "How to Print Floating-Point Numbers Accurately" by Steele and White.
    fn convert_normal_to_string(&self) -> String {
        let (mut integer, mut exp) = self.convert_to_integer();
        let mut buff = Vec::new();
        let ten = BigNum::from_u64(10);

        Self::reduce_printed_integer_length(&mut integer, &mut exp);

        let chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        while !integer.is_zero() {
            let rem = integer.inplace_div(ten);
            let ch = chars[rem.as_u64() as usize];
            buff.insert(0, ch);
        }

        debug_assert!(exp >= 0);
        while buff.len() < exp as usize {
            buff.insert(0, '0');
        }

        buff.insert(buff.len() - exp as usize, '.');
        while !buff.is_empty() && buff[buff.len() - 1] == '0' {
            buff.pop();
        }
        String::from_iter(buff)
    }

    /// Convert the floating point number to a string.
    fn convert_to_string(&self) -> String {
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
impl<const EXPONENT: usize, const MANTISSA: usize> Display
    for Float<EXPONENT, MANTISSA>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.convert_to_string())
    }
}

#[test]
fn test_convert_to_string() {
    use crate::base::FP16;
    use crate::base::FP64;

    fn to_str_w_fp16(val: f64) -> String {
        format!("{}", FP16::from_f64(val))
    }

    fn to_str_w_fp64(val: f64) -> String {
        format!("{}", FP64::from_f64(val))
    }

    assert_eq!("-0.0", to_str_w_fp16(-0.));
    assert_eq!(".3", to_str_w_fp16(0.3));
    assert_eq!("4.5", to_str_w_fp16(4.5));
    assert_eq!("256.", to_str_w_fp16(256.));
    assert_eq!("Inf", to_str_w_fp16(65534.));
    assert_eq!("-Inf", to_str_w_fp16(-65534.));
    assert_eq!(".0999", to_str_w_fp16(0.1));
    assert_eq!(".1", to_str_w_fp64(0.1));
    assert_eq!(".29999999999999998", to_str_w_fp64(0.3));
    assert_eq!("2251799813685248.", to_str_w_fp64((1u64 << 51) as f64));
    assert_eq!("1995.1994999999999", to_str_w_fp64(1995.1995));
}

#[test]
fn test_fuzz_printing() {
    use crate::base::utils;
    use crate::base::FP64;

    let mut lfsr = utils::Lfsr::new();

    for _ in 0..5000 {
        let v0 = lfsr.get64();
        let f0 = f64::from_bits(v0);
        let fp0 = FP64::from_f64(f0);
        fp0.to_string();
    }
}

#[test]
fn test_print_sqrt() {
    type FP = crate::base::FP128;

    // Use Newton-Raphson to find the square root of 5.
    let n = FP::from_u64(5);

    let half = FP::from_u64(2);
    let mut x = n;

    for _ in 0..100 {
        x = half * (x + (n / x));
    }
    println!("{}", x);
}

#[test]
fn test_readme_example() {
    // Create a new type: 15 bits exponent, 112 significand.
    type FP128 = Float<15, 112>;

    // Use Newton-Raphson to find the square root of 5.
    let n = FP128::from_u64(5);

    let two = FP128::from_u64(2);
    let mut x = n;

    for _ in 0..1000 {
        x = (x + (n / x)) / two;
    }
    println!("fp128: {}", x);
    println!("fp64:  {}", x.as_f64());

    let fp = FP16::from_i64(15);
    fp.dump();

}