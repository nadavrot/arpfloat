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

#[derive(Debug, Clone, Copy)]
pub enum RoundMode {
    Trunc,
    Even,
}

/// Round the number in the upper \p num_bits bits of \p val. Zero the rest of
/// the number.
/// xxxxxxxxx becomes xxxxy0000, where y, could be rounded up.
pub fn round_to_even(val: u64, num_bits: usize, mode: RoundMode) -> u64 {
    let rest_bits = 64 - num_bits;
    let is_odd = ((val >> rest_bits) & 0x1) == 1;
    let bottom = val & mask(rest_bits) as u64;
    let half = mask(rest_bits - 1) as u64;

    // Clear the lower part.
    let mut val = (val >> rest_bits) << rest_bits;

    match mode {
        RoundMode::Trunc => val,
        RoundMode::Even => {
            if bottom > half || (bottom == half && is_odd) {
                // If the next few bits are over the half point then round up.
                // Or if the next few bits are exactly half, break the tie and go to even.
                val += 1 << rest_bits;
            }
            val
        }
    }
}

#[test]
fn test_round() {
    assert_eq!(round_to_even(0b101111, 60, RoundMode::Even), 0b110000);
    assert_eq!(round_to_even(0b101111, 60, RoundMode::Trunc), 0b100000);

    let a = 0b100001000001010010110111101011001111001010110000000000000000000;
    let b = 0b100001000001010010110111101011001111001011000000000000000000000;
    assert_eq!(round_to_even(a, 44, RoundMode::Even), b);
}
