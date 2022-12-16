#[derive(Debug, Clone, Copy)]

pub struct Float<const EXPONENT: usize, const SIGNIFICANT: usize> {
    // The sign bit.
    sign: bool,
    // The expotent.
    exp: u64,
    // The significant.
    sig: u64,
}

impl<const EXPONENT: usize, const SIGNIFICANT: usize>
    Float<EXPONENT, SIGNIFICANT>
{
    pub fn new(sign: bool, exp: u64, sig: u64) -> Float<EXPONENT, SIGNIFICANT> {
        Float { sign, exp, sig }
    }
}

#[test]
fn constructor_test() {
    let _: Float<6, 10> = Float::new(false, 10, 123);
}
