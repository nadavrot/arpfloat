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

    pub fn add(a: Self, b: Self, subtract: bool) -> Self {
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
            | (Category::Infinity, Category::NaN) => Self::nan(a.get_sign()),

            (Category::Normal, Category::Infinity)
            | (Category::Zero, Category::Infinity) => {
                Self::inf(b.get_sign() ^ subtract)
            }

            (Category::Zero, Category::Normal) => b,

            (Category::Zero, Category::Zero) => Self::zero(false),

            (Category::Infinity, Category::Infinity) => {
                if a.get_sign() ^ b.get_sign() != subtract {
                    return Self::nan(a.get_sign());
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
    let _ = FP64::add(a, b, false);
}

#[test]
fn test_addition() {
    use super::float::FP64;

    fn add_helper(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        let c = FP64::add(a, b, false);
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
