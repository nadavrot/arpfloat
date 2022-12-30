use super::bigint::BigInt;

use super::float::Float;
use std::fmt::Display;

// Use a bigint for the decimal conversions.
type BigNum = BigInt<6>;

/// \return 5 to the power of \p x: $5^x$.
fn pow5(x: u64) -> BigNum {
    let five = BigNum::from_u64(5);
    let mut v = BigNum::from_u64(1);
    for _ in 0..x {
        v.inplace_mul::<12>(five);
    }
    v
}

#[test]
fn test_pow5() {
    let lookup = [1, 5, 25, 125, 625, 3125, 15625, 78125];
    for (i, val) in lookup.iter().enumerate() {
        assert_eq!(pow5(i as u64).as_u64(), *val);
    }
}

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
                let e5 = pow5((-exp) as u64);
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

    /// This method converts floats to strings. This is a simple implementation
    /// that does not take into account rounding during the round-trip of
    /// parsing-printing of the value, or scientific notation, and the minimal
    /// representation of numbers. For all of that that check out the paper:
    /// "How to Print Floating-Point Numbers Accurately" by Steele and White.
    fn convert_normal_to_string(&self) -> String {
        let (mut integer, exp) = self.convert_to_integer();
        let mut buff = Vec::new();
        let ten = BigNum::from_u64(10);

        let chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        while !integer.is_zero() {
            let rem = integer.inplace_div(ten);
            let ch = chars[rem.as_u64() as usize];
            buff.insert(0, ch);
        }
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
    assert_eq!(".300048828125", to_str_w_fp16(0.3));
    assert_eq!("4.5", to_str_w_fp16(4.5));
    assert_eq!("256.", to_str_w_fp16(256.));
    assert_eq!("Inf", to_str_w_fp16(65534.));
    assert_eq!("-Inf", to_str_w_fp16(-65534.));
    assert_eq!(".0999755859375", to_str_w_fp16(0.1));
    assert_eq!(
        ".1000000000000000055511151231257827021181583404541015625",
        to_str_w_fp64(0.1)
    );
    assert_eq!(
        ".299999999999999988897769753748434595763683319091796875",
        to_str_w_fp64(0.3)
    );
    assert_eq!("2251799813685248.", to_str_w_fp64((1u64 << 51) as f64));
}

#[test]
fn test_print_sqrt() {
    type FP = crate::base::FP128;

    // Use Newton-Raphson to find the square root of some number.
    let n = FP::from_f64(5.);
    let half = FP::from_f64(0.5);
    let mut x = n;

    for _ in 0..100 {
        x = half * (x + (n / x));
    }
    println!("{}", x);
}
