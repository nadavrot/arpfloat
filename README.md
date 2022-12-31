# Arbitrary-Precision Floating-Point Library

ARPFloat is an implementation of arbitrary precision floating point data
structures and utilities. The library can be used to emulate floating point
operation, in software, or create new floating point data types.

### Example
```
#[test]
fn test_readme_example() {
    // Create a new type: 15 bits exponent, 112 significand.
    type FP128 = Float<15, 112>;

    // Create new instances.
    let x = FP128::from_f64(18.3);
    let y = FP128::from_f64(97.32);

    // Do some calculations.
    let z = x+y;
    let val = FP128::mul_with_rm(y, z, RoundingMode::NearestTiesToEven);

    println!("val = {}", val);
}
```

#### License

Licensed under Apache-2.0
