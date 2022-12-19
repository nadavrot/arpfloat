use super::float::Float;

// See Chapter 8. Algorithms for the Five Basic Operations -- Pg 248
pub fn add<const E: usize, const M: usize>(
    x: Float<E, M>,
    y: Float<E, M>,
) -> Float<E, M> {
    if y.get_exp() > x.get_exp() {
        return add(y, x);
    }

    assert!(x.get_exp() >= y.get_exp());
    assert!(!x.in_special_exp() && !y.in_special_exp());

    // Mantissa alignment.
    let exp_delta = x.get_exp() - y.get_exp();
    let mut er = x.get_exp();

    // Addition of the mantissa.

    let y_significand = y.get_mantissa() >> exp_delta.min(63);
    let x_significand = x.get_mantissa();

    let mut is_neg = x.is_negative();

    let is_plus = x.get_sign() == y.get_sign();

    let mut xy_significand;
    if is_plus {
        let res = x_significand.overflowing_add(y_significand);
        xy_significand = res.0;
        if res.1 {
            xy_significand >>= 1;
            xy_significand |= 1 << 63; // Set the implicit bit the overflowed.
            er += 1;
        }
    } else {
        if y_significand > x_significand {
            xy_significand = y_significand - x_significand;
            is_neg ^= true;
        } else {
            xy_significand = x_significand - y_significand;
        }
        // Cancellation happened, we need to normalize the number.
        // Shift xy_significant to the left, and subtract from the exponent
        // until you underflow or until xy_sig is normalized.
        let lz = xy_significand.leading_zeros() as u64;
        let lower_bound = Float::<E, M>::get_exp_bounds().0;
        // How far can we lower the exponent.
        let delta_to_min = er - lower_bound;
        let shift = delta_to_min.min(lz as i64).min(63);
        xy_significand <<= shift;
        er -= shift;
    }

    // Handle the case of cancellation (zero or very close to zero).
    if xy_significand == 0 {
        let mut r = Float::<E, M>::default();
        r.set_mantissa(0);
        r.set_unbiased_exp(0);
        r.set_sign(is_neg);
        return r;
    }

    let mut r = Float::<E, M>::default();
    r.set_mantissa(xy_significand);
    r.set_exp(er);
    r.set_sign(is_neg);
    r
}
#[test]
fn test_addition() {
    use super::float::FP64;

    fn add_helper(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        let c = add(a, b);
        c.as_f64()
    }

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

    while add(add(a, one), a.neg()) == one {
        a = add(a, a);
    }

    let mut b = one.clone();
    while add(add(a, b), a.neg()) != b {
        b = add(b, one);
    }

    assert_eq!(a.as_f64(), 9007199254740992.);
    assert_eq!(b.as_f64(), 2.);
}
