use super::float::{Float, MantissaTy};
use std::fmt::Display;

/// \return 5 to the power of \p x: $5^x$.
fn pow5(x: u64) -> MantissaTy {
    let five = MantissaTy::from_u64(5);
    let mut v = MantissaTy::from_u64(1);
    for _ in 0..x {
        v = v * five;
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
    fn convert_to_integer(&self) -> (MantissaTy, i64) {
        // The natural representation of numbers is 1.mmmmmmm, where the
        // mantissa is aligned to the MSB. In this method we convert the numbers
        // into integers, that start at bit zero, so we use exponent that refers
        //  to bit zero.
        // See Ryu: Fast Float-to-String Conversion -- Ulf Adams.
        // https://youtu.be/kw-U6smcLzk?t=681
        let mut exp = self.get_exp() - MANTISSA as i64;
        let mut mantissa = self.get_mantissa();

        match exp.cmp(&0) {
            std::cmp::Ordering::Less => {
                // The number is not yet an integer, we need to conver it using
                // the method:
                // mmmmm * 5^(e) * 10 ^(-e) == mmmmm * 10 ^ (-e);
                // where (5^e) * (10^-e) == (2^-e)
                // And the left hand side is how we represent our binary number
                // 1.mmmm * 2^-e, and the right-hand-side is how we represent
                // our decimal number 100000 * 10^-e.
                let e5 = pow5((-exp) as u64);
                mantissa = mantissa * e5;
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

    /// This method converts floats to strings. It implements the simple
    /// algorithm that does not take into account rounding during the round-trip
    ///  of parsing-printing of the value, or scientific notation, and the
    /// minimal representation of numbers.  For that check out the paper:
    /// "How to Print Floating-Point Numbers Accurately" by Steele and White.
    fn convert_normal_to_string(&self) -> String {
        let (mut integer, exp) = self.convert_to_integer();
        let mut buff = Vec::new();
        let ten = MantissaTy::from_u64(10);

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
        while buff.len() > 0 && buff[buff.len() - 1] == '0' {
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

    for i in [1.2, 4.5] {
        let n = FP16::from_f64(i as f64);
        let n64: FP64 = n.cast();
        println!("{} vs {} V", n, n64.as_f64());
    }
}
