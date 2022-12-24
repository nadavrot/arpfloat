use super::float::{Category, Float, LossFraction, RoundingMode};

impl<const EXPONENT: usize, const MANTISSA: usize> Float<EXPONENT, MANTISSA> {
    pub fn add_normals(a: Self, b: Self) -> Self {
        assert!(a.get_exp() >= b.get_exp());
        let mut b = b;

        let mut is_neg = a.is_negative();
        let is_plus = a.get_sign() == b.get_sign();

        // Align the mantissa of RHS to have the same alignment as LHS.
        let exp_delta = a.get_exp() - b.get_exp();
        assert!(exp_delta >= 0);
        if exp_delta > 0 {
            b.shift_significand_right(exp_delta as u64);
        }
        assert!(a.get_exp() == b.get_exp());

        // Compute the significand.
        let a_significand = a.get_mantissa();
        let b_significand = b.get_mantissa();
        let ab_significand;

        if is_plus {
            let res = a_significand.overflowing_add(b_significand);
            ab_significand = res.0;
            assert!(!res.1, "Must not overflow!");
        } else if b_significand > a_significand {
            ab_significand = b_significand - a_significand;
            is_neg ^= true;
        } else {
            // Cancellation happened, we need to normalize the number.
            ab_significand = a_significand - b_significand;
        }
        let mut x = Self::new(is_neg, a.get_exp(), ab_significand);
        x.normalize(RoundingMode::NearestTiesToAway, LossFraction::ExactlyZero);
        x
    }

    pub fn add_or_sub_normals(a: Self, b: Self, subtract: bool) -> Self {
        // Make sure that the RHS exponent is larger and eliminate the
        // subtraction by flipping the sign of the RHS.
        if a.absolute_less_than(b) {
            let a = if subtract { a.neg() } else { a };
            return Self::add_normals(b, a);
        }
        let b = if subtract { b.neg() } else { b };
        Self::add_normals(a, b)
    }

    pub fn add(a: Self, b: Self) -> Self {
        Self::add_sub(a, b, false)
    }

    pub fn sub(a: Self, b: Self) -> Self {
        Self::add_sub(a, b, true)
    }

    pub fn add_sub(a: Self, b: Self, subtract: bool) -> Self {
        match (a.get_category(), b.get_category()) {
            (Category::NaN, Category::Infinity)
            | (Category::NaN, Category::NaN)
            | (Category::NaN, Category::Normal)
            | (Category::NaN, Category::Zero)
            | (Category::Normal, Category::Zero)
            | (Category::Infinity, Category::Normal)
            | (Category::Infinity, Category::Zero) => a,

            (Category::Zero, Category::NaN)
            | (Category::Normal, Category::NaN)
            | (Category::Infinity, Category::NaN) => Self::nan(b.get_sign()),

            (Category::Normal, Category::Infinity)
            | (Category::Zero, Category::Infinity) => {
                Self::inf(b.get_sign() ^ subtract)
            }

            (Category::Zero, Category::Normal) => b,

            (Category::Zero, Category::Zero) => Self::zero(false),

            (Category::Infinity, Category::Infinity) => {
                if a.get_sign() ^ b.get_sign() ^ subtract {
                    return Self::nan(a.get_sign() ^ b.get_sign());
                }
                Self::inf(a.get_sign())
            }

            (Category::Normal, Category::Normal) => {
                Self::add_or_sub_normals(a, b, subtract)
            }
        }
    }
}

#[test]
fn test_add() {
    use super::float::FP64;
    let a = FP64::from_u64(1);
    let b = FP64::from_u64(2);
    let _ = FP64::add(a, b);
}

#[test]
fn test_addition() {
    use super::float::FP64;

    fn add_helper(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        let c = FP64::add(a, b);
        c.as_f64()
    }

    assert_eq!(add_helper(0., -4.), -4.);
    assert_eq!(add_helper(-4., 0.), -4.);
    assert_eq!(add_helper(1., 1.), 2.);
    assert_eq!(add_helper(8., 4.), 12.);
    assert_eq!(add_helper(8., 4.), 12.);
    assert_eq!(add_helper(128., 2.), 130.);
    assert_eq!(add_helper(128., -8.), 120.);
    assert_eq!(add_helper(64., -60.), 4.);
    assert_eq!(add_helper(69., -65.), 4.);
    assert_eq!(add_helper(69., 69.), 138.);
    assert_eq!(add_helper(69., 1.), 70.);
    assert_eq!(add_helper(-128., -8.), -136.);
    assert_eq!(add_helper(64., -65.), -1.);
    assert_eq!(add_helper(-64., -65.), -129.);
    assert_eq!(add_helper(-15., -15.), -30.);

    assert_eq!(add_helper(-15., 15.), 0.);

    for i in -4..15 {
        for j in i..15 {
            assert_eq!(
                add_helper(f64::from(j), f64::from(i)),
                f64::from(i) + f64::from(j)
            );
        }
    }
}

// Pg 120.  Chapter 4. Basic Properties and Algorithms.
#[test]
fn test_addition_large_numbers() {
    use super::float::FP64;

    let one = FP64::from_i64(1);
    let mut a = FP64::from_i64(1);

    while FP64::sub(FP64::add(a, one), a) == one {
        a = FP64::add(a, a);
    }

    let mut b = one;
    while FP64::sub(FP64::add(a, b), a) != b {
        b = FP64::add(b, one);
    }

    assert_eq!(a.as_f64(), 9007199254740992.);
    assert_eq!(b.as_f64(), 2.);
}

#[test]
fn add_denormals() {
    use super::float::FP64;

    let v0 = f64::from_bits(0x0000_0000_0010_0010);
    let v1 = f64::from_bits(0x0000_0000_1001_0010);
    let v2 = f64::from_bits(0x1000_0000_0001_0010);

    let a0 = FP64::from_f64(v0);
    assert_eq!(a0.as_f64(), v0);

    fn add_f64(a: f64, b: f64) -> f64 {
        let a0 = FP64::from_f64(a);
        let b0 = FP64::from_f64(b);
        assert_eq!(a0.as_f64(), a);
        FP64::add(a0, b0).as_f64()
    }

    // Add and subtract denormals.
    assert_eq!(add_f64(v0, v1), v0 + v1);
    assert_eq!(add_f64(v0, -v0), v0 - v0);
    assert_eq!(add_f64(v0, v2), v0 + v2);
    assert_eq!(add_f64(v2, v1), v2 + v1);
    assert_eq!(add_f64(v2, -v1), v2 - v1);

    // Add and subtract denormals and normal numbers.
    assert_eq!(add_f64(v0, 10.), v0 + 10.);
    assert_eq!(add_f64(v0, -10.), v0 - 10.);
    assert_eq!(add_f64(10000., v0), 10000. + v0);
}

#[test]
fn add_special_values() {
    use crate::base::utils;

    // Test the addition of various irregular values.
    let values = utils::get_special_test_values();

    use super::float::FP64;

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        FP64::add(a, b).as_f64()
    }

    for v0 in values {
        for v1 in values {
            let r0 = add_f64(v0, v1);
            let r1 = v0 + v1;
            assert_eq!(r0.is_finite(), r1.is_finite());
            assert_eq!(r0.is_nan(), r1.is_nan());
            assert_eq!(r0.is_infinite(), r1.is_infinite());
            let r0_bits = r0.to_bits();
            let r1_bits = r1.to_bits();
            println!("{}  + {}", v0, v1);
            println!("{} vs {}", r0, r1);
            println!("|{:64b} X vs \n|{:64b} V", r0_bits, r1_bits);
            // Check that the results are bit identical, or are both NaN.
            assert!(!r0.is_nan() || r0_bits == r1_bits);
        }
    }
}

#[test]
fn test_simple() {
    use super::float::FP64;

    let a: f64 = 24.0;
    let b: f64 = 0.1;

    println!("Input");
    println!("| {:64b} X  ({})", a.to_bits(), a);
    println!("| {:64b} V  ({})", b.to_bits(), b);

    let af = FP64::from_f64(a);
    let bf = FP64::from_f64(b);
    let cf = FP64::add(af, bf);

    let r0 = cf.as_f64();
    let r1: f64 = 24.0f64 + 0.1f64;

    println!("Output");
    println!("| {:64b} X  ({})", r0.to_bits(), r0);
    println!("| {:64b} V  ({})", r1.to_bits(), r1);

    println!("What happened:");
    af.dump();
    bf.dump();
    cf.dump();
}
