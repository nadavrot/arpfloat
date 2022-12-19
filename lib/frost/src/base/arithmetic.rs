use super::float::Float;

/// \returns (\p x - \p y).
pub fn sub<const E: usize, const M: usize>(
    x: Float<E, M>,
    y: Float<E, M>,
) -> Float<E, M> {
    add(x, y.neg())
}

/// \returns (\p x + \p y).
// See Chapter 8. Algorithms for the Five Basic Operations -- Pg 248
pub fn add<const E: usize, const M: usize>(
    x: Float<E, M>,
    y: Float<E, M>,
) -> Float<E, M> {
    if y.get_exp() > x.get_exp() {
        return add(y, x);
    }

    let same_sign = x.get_sign() != y.get_sign();

    // 8.3. Floating-Point Addition and Subtraction 247
    if x.is_nan() {
        return Float::<E, M>::nan(x.get_sign());
    }
    if y.is_nan() {
        return Float::<E, M>::nan(y.get_sign());
    }
    if x.is_inf() {
        if y.is_inf() && same_sign {
            return Float::<E, M>::nan(true);
        }
        return Float::<E, M>::inf(x.get_sign());
    }
    if y.is_inf() {
        if x.is_inf() && same_sign {
            return Float::<E, M>::nan(true);
        }
        return Float::<E, M>::inf(y.get_sign());
    }

    if x.is_zero() && y.is_zero() {
        return Float::<E, M>::zero(x.get_sign() && y.get_sign());
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
        return Float::<E, M>::zero(false);
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

    while sub(add(a, one), a) == one {
        a = add(a, a);
    }

    let mut b = one;
    while sub(add(a, b), a) != b {
        b = add(b, one);
    }

    assert_eq!(a.as_f64(), 9007199254740992.);
    assert_eq!(b.as_f64(), 2.);
}

#[test]
fn add_denormals() {
    use super::float::FP64;

    let v0 = f64::from_bits(0x0000_0000_0010_0001);
    let v1 = f64::from_bits(0x0000_0000_1001_0001);
    let v2 = f64::from_bits(0x1000_0000_0001_0001);

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        add(a, b).as_f64()
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
    // Test the addition of different irregular values.
    let values = [
        -f64::NAN,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        10.,
        -10.,
    ];
    use super::float::FP64;

    fn add_f64(a: f64, b: f64) -> f64 {
        let a = FP64::from_f64(a);
        let b = FP64::from_f64(b);
        add(a, b).as_f64()
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
            // Check that the results are bit identical, or are both NaN.
            assert!(!r0.is_nan() || r0_bits == r1_bits);
        }
    }
}
