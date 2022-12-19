/// \returns a mask full of 1s, of \p b bits.
pub fn mask(b: usize) -> usize {
    (1 << (b)) - 1
}

#[test]
fn test_masking() {
    assert_eq!(mask(0), 0x0);
    assert_eq!(mask(1), 0x1);
    assert_eq!(mask(8), 255);
}

/// Convert a mantissa in the implicit format (no possible leading 1 bit) to
/// the internal storage format. If \p leading_1 is set then a leading one is
/// added (otherwise it is a subnormal).
/// Format: [1 IIIIII 00000000]
pub fn expand_mantissa_to_explicit<const FROM: usize>(
    input: u64,
    leading_1: bool,
) -> u64 {
    let value: u64 = if leading_1 { 1 << 63 } else { 0 };
    let shift = 63 - FROM;
    value | (input << shift)
}

#[test]
fn test_expand_mantissa() {
    assert_eq!(expand_mantissa_to_explicit::<8>(0, true), 1 << 63);
    assert_eq!(
        expand_mantissa_to_explicit::<8>(1, true),
        0x8080000000000000
    );
    assert_eq!(
        expand_mantissa_to_explicit::<32>(0xffffffff, false),
        0x7fffffff80000000
    );
}
